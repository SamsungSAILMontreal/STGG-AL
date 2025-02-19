import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


import torch.distributed as dist
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger

import moses
from moses.utils import disable_rdkit_log, enable_rdkit_log
from pytorch_lightning.utilities import rank_zero_only

from model.generator import BaseGenerator
from data.dataset import ZincDataset, MosesDataset, SimpleMosesDataset, QM9Dataset
from data.target_data import Data
from utils.utils import compute_sequence_accuracy, compute_sequence_cross_entropy, canonicalize
#from evaluate.nspdk import nspdk_stats
#from data.smiles import mols_to_nx

class EmptyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 16

    def __getitem__(self, index):
        return torch.zeros(16)

class BaseGeneratorLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseGeneratorLightningModule, self).__init__()
        hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_datasets()
        self.setup_model()
        
    def setup_datasets(self, hparams):
        dataset_cls = {
            "zinc": ZincDataset,
            "moses": MosesDataset,
            "simplemoses": SimpleMosesDataset,
            "qm9": QM9Dataset,
        }.get(hparams.dataset_name)
        self.train_dataset = dataset_cls(split="train", randomize_order=hparams.randomize_order, MAX_LEN=hparams.max_len)
        self.val_dataset = dataset_cls(split="valid", randomize_order=hparams.randomize_order, MAX_LEN=hparams.max_len)
        self.test_dataset = dataset_cls(split="test", randomize_order=hparams.randomize_order, MAX_LEN=hparams.max_len)
        self.train_smiles_set = set(self.train_dataset.smiles_list)

    def setup_model(self, hparams):
        self.model = BaseGenerator(
            num_layers=hparams.num_layers,
            emb_size=hparams.emb_size,
            nhead=hparams.nhead,
            dim_feedforward=hparams.dim_feedforward,
            input_dropout=hparams.input_dropout,
            dropout=hparams.dropout,
            disable_treeloc=hparams.disable_treeloc,
            disable_graphmask=hparams.disable_graphmask, 
            disable_valencemask=hparams.disable_valencemask,
            enable_absloc=hparams.enable_absloc,
            properties_as_tokens=False,
            MAX_LEN=hparams.max_len,
        )

    ### Dataloaders and optimizers
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=Data.collate,
            num_workers=self.hparams.num_workers,
            drop_last=True, 
            persistent_workers=True, pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=Data.collate,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            persistent_workers=True, pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            EmptyDataset(),
            batch_size=0,
            shuffle=False,
            num_workers=0,
        )

    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay,
            )
        # Warmup + Cosine scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=self.hparams.warmup_steps,
        )
        n_iters_per_epoch = -(-len(self.train_dataset) // self.hparams.batch_size) # because drop_last=False we must round up
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_iters_per_epoch*self.trainer.fit_loop.max_epochs - self.hparams.warmup_steps + 1,
            eta_min=self.hparams.lr*self.hparams.lr_decay
        )
        combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.hparams.warmup_steps])
        lr_scheduler = {
            "scheduler": combined_scheduler,
            'interval': 'step'
        }
        return [optimizer], [lr_scheduler]

    ### Main steps
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()

        # decoding
        logits = self.model(batched_data)
        loss, _ = compute_sequence_cross_entropy(logits, batched_data[0], ignore_index=0)
        statistics["loss/total"] = loss
        statistics["acc/total"] = compute_sequence_accuracy(logits, batched_data[0], ignore_index=0)[0]

        return loss, statistics

    def training_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True, sync_dist=self.hparams.n_gpu > 1)
        return loss

    def validation_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        for key, val in statistics.items():
            self.log(f"validation/{key}", val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)
        return loss

    def test_step(self, batched_data, batch_idx):
        return torch.zeros(1)

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.hparams.check_sample_every_n_epoch == 0:
            self.check_samples()

    def on_test_epoch_end(self):
        self.check_samples()

    def check_samples(self):
        assert self.hparams.num_samples % self.hparams.n_gpu == 0
        num_samples = self.hparams.num_samples // self.hparams.n_gpu if not self.trainer.sanity_checking else 2
        local_smiles_list, results = self.sample(num_samples)

        # Save molecules
        #if not self.trainer.sanity_checking:
        #    for (smiles, result) in zip(smiles_list, results):
        #        self.logger.experiment[f"sample/smiles/{self.current_epoch:03d}"].log(smiles)
        #        self.logger.experiment[f"sample/result/{self.current_epoch:03d}"].log(result)

        # Gather results
        if self.hparams.n_gpu > 1:
            global_smiles_list = [None for _ in range(self.hparams.n_gpu)]
            dist.all_gather_object(global_smiles_list, local_smiles_list)
            smiles_list = []
            for i in range(self.hparams.n_gpu):
                smiles_list += global_smiles_list[i]
        else:
            smiles_list = local_smiles_list

        #
        valid_smiles_list = [smiles for smiles in smiles_list if smiles is not None]
        unique_smiles_set = set(valid_smiles_list)
        novel_smiles_list = [smiles for smiles in valid_smiles_list if smiles not in self.train_smiles_set]
        statistics = dict()
        statistics["sample/valid"] = float(len(valid_smiles_list)) / self.hparams.num_samples
        statistics["sample/unique"] = float(len(unique_smiles_set)) / len(valid_smiles_list)
        statistics["sample/novel"] = float(len(novel_smiles_list)) / len(valid_smiles_list)

        #
        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)

        #nspdk_score = nspdk_stats(self.test_graph_list, mols_to_nx(gen_mols))
        #self.log("sample/nspdk", nspdk_score, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)
        
        if len(valid_smiles_list) > 0:
            torch.backends.cudnn.enabled = False
            moses_statistics = moses.get_all_metrics(
                smiles_list, 
                n_jobs=self.hparams.num_workers,#*self.hparams.n_gpu, 
                device=str(self.device), 
                train=self.train_dataset.smiles_list, 
                test=self.test_dataset.smiles_list,
            )
            for key in moses_statistics:
                self.log(f"sample/moses/{key}", moses_statistics[key], on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)#, rank_zero_only=True)
            torch.backends.cudnn.enabled = True

    def sample(self, num_samples):
        offset = 0
        results = []
        self.model.eval()
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples
            #print(offset)
            data_list = self.model.decode(cur_num_samples, max_len=self.hparams.max_len, device=self.device)
            results.extend((data.to_smiles(), "".join(data.tokens), data.error) for data in data_list)
        self.model.train()
        
        disable_rdkit_log()
        smiles_list = [canonicalize(elem[0]) for elem in results]
        enable_rdkit_log()

        return smiles_list, results

    @staticmethod
    def add_args(parser):
        #
        parser.add_argument("--dataset_name", type=str, default="zinc")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=6)
        parser.add_argument("--randomize_order", action="store_true") # randomize order of nodes and edges for the spanning tree to produce more diversity

        #
        parser.add_argument("--num_layers", type=int, default=6)
        parser.add_argument("--emb_size", type=int, default=1024)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--input_dropout", type=float, default=0.0)
        parser.add_argument("--dropout", type=float, default=0.1) # 0.0 for llms
        parser.add_argument("--logit_hidden_dim", type=int, default=256)

        #
        parser.add_argument("--disable_treeloc", action="store_true")
        parser.add_argument("--disable_graphmask", action="store_true")
        parser.add_argument("--disable_valencemask", action="store_true")
        parser.add_argument("--enable_absloc", action="store_true")
        
        #
        parser.add_argument("--lr", type=float, default=1e-4) # varies for llms
        parser.add_argument("--warmup_steps", type=int, default=0) # 200-1k should be good
        parser.add_argument("--lr_decay", type=float, default=1.0) # 0.1 for llms
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.999) # 0.95 for llms
        parser.add_argument("--weight_decay", type=float, default=0.0) # 0.1 for llms

        #
        parser.add_argument("--max_len", type=int, default=200)
        parser.add_argument("--check_sample_every_n_epoch", type=int, default=10)
        parser.add_argument("--save_every_n_epoch", type=int, default=5) # how often to save checkpoints
        parser.add_argument("--num_samples", type=int, default=10000) 
        parser.add_argument("--sample_batch_size", type=int, default=2500)

        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BaseGeneratorLightningModule.add_args(parser)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--save_checkpoint_dir", type=str, default="CHANGE_TO_YOUR_DIR/AutoregressiveMolecules_checkpoints")
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--test", action="store_true")
    hparams = parser.parse_args()

    print('Warning: Note that for both training and metrics, results will only be reproducible when using the same number of GPUs and num_samples/sample_batch_size')
    pl.seed_everything(hparams.seed, workers=True) # use same seed, except for the dataloaders
    model = BaseGeneratorLightningModule(hparams)
    if hparams.load_checkpoint_path != "":
        model.load_state_dict(torch.load(hparams.load_checkpoint_path)["state_dict"])
    if hparams.compile:
        model = torch.compile(model)

    neptune_logger = NeptuneLogger(
        api_key="YOUR_API_KEY",
        project="YOUR_PROJECT_KEY",
        source_files="**/*.py",
        tags=hparams.tag.split("_")
        )
    neptune_logger.log_hyperparams(vars(hparams))
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(hparams.save_checkpoint_dir, hparams.tag), 
        monitor="validation/loss/total",
        save_top_k=1, 
        mode="min",
        save_last=True,
        every_n_epochs=hparams.save_every_n_epoch,
        enable_version_counter=False,
    )
    trainer = pl.Trainer(
        devices=hparams.n_gpu, 
        accelerator="cpu" if hparams.cpu else "gpu",
        strategy="ddp" if hparams.n_gpu > 1 else 'auto',
        precision="bf16-mixed" if hparams.bf16 else "32-true",
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=hparams.gradient_clip_val,
    )
    pl.seed_everything(hparams.seed + trainer.global_rank, workers=True) # different seed per worker
    trainer.fit(model, 
            ckpt_path='last')
    pl.seed_everything(hparams.seed + trainer.global_rank)
    if hparams.test:
        trainer.test(model, 
            ckpt_path=checkpoint_callback.best_model_path)