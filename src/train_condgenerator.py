import os
import sys
import argparse
from numpy.lib.arraysetops import unique
from rdkit import Chem
import random
import torch
import copy
from torch.utils.data import DataLoader
import shutil
import datetime

import torch.distributed as dist
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
import moses
from moses.utils import disable_rdkit_log, enable_rdkit_log
from data.target_data import Data as TargetData
from torch.distributions.bernoulli import Bernoulli

from model.generator import CondGenerator
from data.dataset import get_cond_datasets, DummyDataset, merge_datasets
from props.properties import penalized_logp, MAE_properties, MAE_properties_estimated, best_rewards_gflownet, get_xtb_scores, return_top_k, remove_duplicates, average_similarity, average_similarity2, average_similarity_
from utils.utils import compute_sequence_cross_entropy, compute_property_accuracy, compute_sequence_accuracy, canonicalize
from train_generator import BaseGeneratorLightningModule
from evaluate.MCD.evaluator import TaskModel, compute_molecular_metrics
from moses.metrics.metrics import compute_intermediate_statistics
from joblib import dump, load
import numpy as np

def max_ring_size(mol):
    max_size_Rings = 0
    ri = mol.GetRingInfo()
    for ring in ri.BondRings():
        max_size_Rings = max(max_size_Rings, len(ring))
    return max_size_Rings

class CondGeneratorLightningModule(BaseGeneratorLightningModule):
    def __init__(self, hparams):
        super(CondGeneratorLightningModule, self).__init__(hparams)
        
    def setup_datasets(self):
        raw_dir = f"../resource/data/{self.hparams.dataset_name}"
        self.train_dataset, self.val_dataset, self.test_dataset = get_cond_datasets(dataset_name=self.hparams.dataset_name,
            raw_dir=raw_dir, randomize_order=self.hparams.randomize_order, 
            MAX_LEN=self.hparams.max_len, scaling_type=self.hparams.scaling_type, 
            gflownet=self.hparams.gflownet, gflownet_realgap=self.hparams.gflownet_realgap, 
            start_min=not self.hparams.start_random, sort=self.hparams.legacy_sort,
            finetune_dataset=self.hparams.extra_dataset_name if self.hparams.pretrain else None, 
            pretrain_dataset=self.hparams.extra_dataset_name if self.hparams.finetune else None, 
            limited_properties=self.hparams.limited_properties, 
            extra_props_dont_always_mask=self.hparams.extra_props_dont_always_mask, mask_seperately=self.hparams.mask_seperately,
            mask_props=self.hparams.mask_prop_in_original,
            remove_properties=self.hparams.remove_properties,
            portion_used=self.hparams.portion_used,
            load_generated_mols=self.hparams.load_generated_mols,
            remove_unseen=not self.hparams.dont_remove_unseen_tokens,
            force_prop_redo=self.hparams.force_prop_redo,
            reward_type=self.hparams.xtb_reward_type)

        # Add an extra data dataset
        if self.hparams.extra_dataset_name != '':
            raw_dir2 = f"../resource/data/{self.hparams.extra_dataset_name}"
            train_dataset2, val_dataset2, test_dataset2 = get_cond_datasets(dataset_name=self.hparams.extra_dataset_name,
                raw_dir=raw_dir2, randomize_order=self.hparams.randomize_order, 
                MAX_LEN=self.hparams.max_len, scaling_type=self.hparams.scaling_type, 
                gflownet=self.hparams.gflownet, gflownet_realgap=self.hparams.gflownet_realgap, 
                start_min=not self.hparams.start_random,
                finetune_dataset=self.hparams.dataset_name if self.hparams.finetune else None, 
                pretrain_dataset=self.hparams.dataset_name if self.hparams.pretrain else None, 
                limited_properties=self.hparams.limited_properties, 
                extra_props_dont_always_mask=self.hparams.extra_props_dont_always_mask, mask_seperately=self.hparams.mask_seperately,
                mask_props=self.hparams.mask_prop_in_extra,
                remove_properties=self.hparams.remove_properties,
                load_generated_mols=self.hparams.load_generated_mols_extra,
                remove_unseen=not self.hparams.dont_remove_unseen_tokens,
                force_prop_redo=self.hparams.force_prop_redo,
                reward_type=self.hparams.xtb_reward_type)

            # Combined dataset
            train_dataset_combined = merge_datasets(self.train_dataset, train_dataset2)
            val_dataset_combined = merge_datasets(self.val_dataset, val_dataset2, scaler_std_properties=train_dataset_combined.scaler_std_properties, scaler_properties=train_dataset_combined.scaler_properties)
            test_dataset_combined = merge_datasets(self.test_dataset, test_dataset2, scaler_std_properties=train_dataset_combined.scaler_std_properties, scaler_properties=train_dataset_combined.scaler_properties)

            # In some cases, we want the scaler for the original or the extra dataset
            if self.hparams.scaler_vocab == 'original':
                train_dataset_combined.update_scalers(self.train_dataset)
                val_dataset_combined.update_scalers(self.val_dataset)
                test_dataset_combined.update_scalers(self.test_dataset)
            elif self.hparams.scaler_vocab == 'extra':
                train_dataset_combined.update_scalers(test_dataset2)
                val_dataset_combined.update_scalers(val_dataset2)
                test_dataset_combined.update_scalers(test_dataset2)
            else:
                pass

            if self.hparams.finetune or self.hparams.pretrain: # combine both datasets vocabs and ensure good scaler, but keep the molecules and properties from the original dataset
                self.train_dataset.update_vocabs_scalers(train_dataset_combined)
                self.val_dataset.update_vocabs_scalers(val_dataset_combined)
                self.test_dataset.update_vocabs_scalers(test_dataset_combined)
            else: # combine both datasets
                self.train_dataset = train_dataset_combined
                self.val_dataset = val_dataset_combined
                self.test_dataset = test_dataset_combined

        print(f"--Atom vocabulary size--: {len(self.train_dataset.vocab.ATOM_TOKENS)}")
        print(self.train_dataset.vocab.ATOM_TOKENS)
        print(self.train_dataset.properties.shape)

        self.train_smiles_set = set(self.train_dataset.smiles_list)
        self.hparams.vocab = self.train_dataset.vocab

        def collate_fn(data_list):
            batched_mol_data, batched_cond_data = zip(*data_list)
            return TargetData.collate(batched_mol_data), torch.stack(batched_cond_data, dim=0)    
        self.collate_fn = collate_fn

        self.hparams.n_properties = self.train_dataset.properties.shape[1]
        self.hparams.cat_var_index = self.train_dataset.categorical_prop
        self.hparams.cont_var_index = self.train_dataset.continuous_prop

        self.mask_prop = self.train_dataset.mask_prop
        self.extra_prop = self.train_dataset.extra_prop
        if self.mask_prop is not None:
            self.hparams.cont_var_index = list(set(self.hparams.cont_var_index) - set(np.where(self.mask_prop)[0].tolist()))

    def setup_model(self):
        self.model = CondGenerator(
            num_layers=self.hparams.num_layers,
            emb_size=self.hparams.emb_size,
            nhead=self.hparams.nhead,
            dim_feedforward=self.hparams.dim_feedforward,
            input_dropout=self.hparams.input_dropout,
            dropout=self.hparams.dropout,
            disable_treeloc=self.hparams.disable_treeloc,
            disable_graphmask=self.hparams.disable_graphmask, 
            disable_valencemask=self.hparams.disable_valencemask,
            disable_counting_ring=self.hparams.disable_counting_ring,
            disable_random_prop_mask=self.hparams.disable_random_prop_mask,
            enable_absloc=self.hparams.enable_absloc,
            lambda_predict_prop=self.hparams.lambda_predict_prop,
            MAX_LEN=self.hparams.max_len,
            gpt=self.hparams.gpt,
            bias=not self.hparams.no_bias,
            rotary=self.hparams.rotary,
            rmsnorm=self.hparams.rmsnorm,
            swiglu=self.hparams.swiglu,
            expand_scale=self.hparams.expand_scale,
            special_init=self.hparams.special_init,
            n_properties=self.hparams.n_properties,
            vocab=self.hparams.vocab,
            n_correct=self.hparams.n_correct,
            cond_lin=self.hparams.cond_lin,
            cat_var_index=self.hparams.cat_var_index,
            cont_var_index=self.hparams.cont_var_index,
            bin_loss_type=self.hparams.bin_loss_type,
        )

    ### Dataloaders and optimizers
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=0 if self.hparams.test else self.hparams.num_workers,
            drop_last=False,
            persistent_workers=not self.hparams.test and self.hparams.num_workers > 0, pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0 if self.hparams.test else self.hparams.num_workers,
            drop_last=False,
            persistent_workers=not self.hparams.test and self.hparams.num_workers > 0, pin_memory=True,
        )

    def test_dataloader(self):
        self.test_step_pred_acc = []
        if self.hparams.no_test_step:
            return DataLoader(
                DummyDataset(),
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )
        elif self.hparams.test_on_train_data:
            dset = self.train_dataset
        else:
            dset = self.test_dataset
        return DataLoader(
            dset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers if self.hparams.test else 0,
            drop_last=False,
            persistent_workers=False, pin_memory=True,
        )

    def shared_step(self, batched_data, test=False):
        loss, statistics = 0.0, dict()

        # decoding
        batched_mol_data, batched_cond_data = batched_data
        logits, pred_prop = self.model(batched_mol_data, batched_cond_data, mask_cond=self.mask_prop, extra_prop=self.extra_prop)
        loss, loss_prop_cont, loss_prop_cat = compute_sequence_cross_entropy(logits, batched_mol_data[0], ignore_index=0, 
            prop=batched_cond_data, pred_prop=pred_prop, lambda_predict_prop=self.hparams.lambda_predict_prop, 
            lambda_predict_prop_always=self.hparams.lambda_predict_prop_always, 
            cont_var_index=self.hparams.cont_var_index, cat_var_index=self.hparams.cat_var_index,
            bin_loss_type=self.hparams.bin_loss_type)

        statistics["loss/total"] = loss
        statistics["loss/class"] = loss - loss_prop_cont - loss_prop_cat
        statistics["loss/prop_cont"] = loss_prop_cont
        statistics["loss/prop_cat"] = loss_prop_cat
        statistics["acc/total"] = compute_sequence_accuracy(logits, batched_mol_data[0], ignore_index=0)[0]

        if self.hparams.lambda_predict_prop > 0 and not (loss_prop_cat == 0.0 and loss_prop_cont == 0.0):
            if test:
                b = batched_mol_data[0].shape[0]
                test_step_pred_acc = compute_property_accuracy(batched_mol_data[0], 
                    prop=batched_cond_data, pred_prop=pred_prop, 
                    cont_var_index=self.hparams.cont_var_index, cat_var_index=self.hparams.cat_var_index, mean=False)
                self.test_step_pred_acc.append(test_step_pred_acc)
            else:
                validation_step_pred_acc = compute_property_accuracy(batched_mol_data[0], 
                    prop=batched_cond_data, pred_prop=pred_prop, 
                    cont_var_index=self.hparams.cont_var_index, cat_var_index=self.hparams.cat_var_index, mean=True)
                for i, prop_acc in enumerate(validation_step_pred_acc):
                    statistics[f"acc/prop{i}"] = prop_acc
        return loss, statistics

    def training_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True, sync_dist=self.hparams.n_gpu > 1)
        return loss

    def validation_step(self, batched_data, batch_idx):
        if self.hparams.no_test_step:
            loss = 0.0
        else:
            loss, statistics = self.shared_step(batched_data)
            for key, val in statistics.items():
                self.log(f"validation/{key}", val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)
        return loss

    def test_step(self, batched_data, batch_idx):
        if self.hparams.no_test_step:
            loss = 0.0
        else:
            loss, statistics = self.shared_step(batched_data, test=not self.hparams.no_test_set_accuracy)
            if self.hparams.lambda_predict_prop > 0 and not self.hparams.no_test_set_accuracy:
                test_step_pred_acc = torch.cat(self.test_step_pred_acc, dim=0) # [N, p_variables]
                if self.hparams.n_gpu > 1:
                    all_test_step_pred_acc = self.all_gather(test_step_pred_acc).view(-1, self.hparams.n_properties)
                else:
                    all_test_step_pred_acc = test_step_pred_acc
                all_test_step_pred_acc = all_test_step_pred_acc.mean(dim=0) # [p_variables]
                self.test_step_pred_acc.clear()
                for i, prop_acc in enumerate(all_test_step_pred_acc):
                    statistics[f"acc/prop{i}"] = prop_acc
            for key, val in statistics.items():
                self.log(f"validation/{key}", val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)
        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking and not self.hparams.test and self.current_epoch > 0: # can lead to valence errors when untrained due to bad choices
            if (self.current_epoch + 1) % self.hparams.check_sample_every_n_epoch == 0:
                if self.hparams.specific_conditioning:
                    self.check_samples_specific()
                elif self.hparams.gflownet:
                    self.check_samples_gflownet()
                elif self.hparams.graph_dit_benchmark:
                    assert not self.hparams.ranking_based_best_of_k # We have different properties, do not use ranking
                    self.check_samples_dit() # cond on train properties with special metrics
                else:
                    if not self.hparams.ranking_based_best_of_k and not self.hparams.only_ood: # We have different properties, do not use ranking
                        self.check_samples_uncond() # uncond
                    if not self.hparams.no_ood: 
                        self.check_samples_ood() # cond on ood values
                    else:
                        self.check_samples() # cond on train properties

    def on_test_epoch_end(self):
        if len(self.hparams.check_sim) > 0:
            self.check_sim()
        elif not self.trainer.sanity_checking and self.current_epoch > 0: # can lead to valence errors when untrained due to bad choices
            if self.hparams.specific_conditioning:
                self.check_samples_specific()
            elif self.hparams.gflownet:
                self.check_samples_gflownet()
            elif self.hparams.graph_dit_benchmark:
                assert not self.hparams.ranking_based_best_of_k # We have different properties, do not use ranking
                self.check_samples_dit() # cond on train properties with special metrics
            else:
                if not self.hparams.ranking_based_best_of_k and not self.hparams.only_ood: # We have different properties, do not use ranking
                    self.check_samples_uncond() # uncond
                if not self.hparams.no_ood: 
                    self.check_samples_ood() # cond on ood values
                else:
                    self.check_samples() # cond on train properties

    # Sample extreme values for each properties and evaluate the OOD generated molecules 
    def check_samples_ood(self):
        assert self.hparams.num_samples_ood % self.hparams.n_gpu == 0
        num_samples = self.hparams.num_samples_ood // self.hparams.n_gpu if not self.trainer.sanity_checking else 2
        assert len(self.hparams.ood_values) == 1 or len(self.hparams.ood_values) == 2*len(self.hparams.ood_names)
        i_ = -1
        
        for feat_name in self.hparams.ood_names:
            idx = np.where(feat_name == self.train_dataset.properties_names)[0][0]
            for j in range(2): #
                i_ += 1
                if self.hparams.ood_values[i_] == 666:
                    break
                if j == 0: # u + std
                    stats_name = f"sample_ood_{idx}_{feat_name}_mean_plus_std"
                    if len(self.hparams.ood_values) <= 1:
                        properties_np = self.train_dataset.get_mean_plus_std_property(idx, std=4)
                        print(f'raw_prop: {self.train_dataset.scaler_properties.inverse_transform(properties_np)[0]}')
                    else:
                        properties_np = np.zeros((1, self.train_dataset.n_properties))
                        properties_np[:, idx] = self.hparams.ood_values[i_]
                        print(f'raw_prop: {properties_np[0]}')
                        if self.train_dataset.scaler_properties is not None:
                            properties_np = self.train_dataset.scaler_properties.transform(properties_np) # raw to whatever
                else: # u - std
                    stats_name = f"sample_ood_{idx}_{feat_name}_mean_minus_std"
                    if len(self.hparams.ood_values) <= 1:
                        properties_np = self.train_dataset.get_mean_plus_std_property(idx, std=-4)
                        print(f'raw_prop: {self.train_dataset.scaler_properties.inverse_transform(properties_np)[0]}')
                    else:
                        properties_np = np.zeros((1, self.train_dataset.n_properties))
                        properties_np[:, idx] = self.hparams.ood_values[i_]
                        print(f'raw_prop: {properties_np[0]}')
                        if self.train_dataset.scaler_properties is not None:
                            properties_np = self.train_dataset.scaler_properties.transform(properties_np) # raw to whatever
                print(f'std_prop: {properties_np[0, idx]}')
                print(stats_name)
                properties_np = np.repeat(properties_np, num_samples, axis=0)
                local_properties = torch.tensor(properties_np).to(device=self.device, dtype=torch.float32)
                mask_cond = [i != idx for i in range(self.train_dataset.n_properties)] # we only give the single property we care about to the model; the model can figure out the rest
                local_smiles_list, results, local_prop_pred = self.sample_cond(local_properties, num_samples, 
                    temperature_min=self.hparams.temperature_min, temperature_max=self.hparams.temperature_max, 
                    guidance_min=self.hparams.guidance_min, guidance_max=self.hparams.guidance_max, mask_cond=mask_cond)
                #print('smiles')
                #print(local_smiles_list[0:5])

                # Gather results
                if self.hparams.n_gpu > 1:
                    global_smiles_list = [None for _ in range(self.hparams.n_gpu)]
                    dist.all_gather_object(global_smiles_list, local_smiles_list)
                    smiles_list = []
                    for i in range(self.hparams.n_gpu):
                        smiles_list += global_smiles_list[i]

                    global_properties = [torch.zeros_like(local_properties) for _ in range(self.hparams.n_gpu)]
                    dist.all_gather(global_properties, local_properties)
                    properties = torch.cat(global_properties, dim=0)

                    if local_prop_pred is not None:
                        global_prop_pred = [torch.zeros_like(local_prop_pred) for _ in range(self.hparams.n_gpu)]
                        dist.all_gather(global_prop_pred, local_prop_pred)
                        prop_pred = torch.cat(global_prop_pred, dim=0)
                    else:
                        prop_pred = None
                else:
                    smiles_list = local_smiles_list
                    properties = local_properties
                    prop_pred = local_prop_pred

                idx_valid = []
                valid_smiles_list = []
                valid_mols_list = []
                for smiles in smiles_list:
                    if smiles is not None:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None and mol.GetNumHeavyAtoms() <= self.hparams.max_number_of_atoms and max_ring_size(mol) <= self.hparams.max_ring_size:
                            idx_valid += [True]
                            valid_smiles_list += [smiles]
                            valid_mols_list += [mol]
                        else:
                            idx_valid += [False]
                    else:
                        idx_valid += [False]
                unique_smiles_set = set(valid_smiles_list)
                novel_smiles_list = [smiles for smiles in valid_smiles_list if smiles not in self.train_smiles_set]
                efficient_smiles_list = [smiles for smiles in unique_smiles_set if smiles in novel_smiles_list]
                statistics = dict()

                statistics[f"{stats_name}/valid"] = float(len(valid_smiles_list)) / self.hparams.num_samples_ood
                statistics[f"{stats_name}/unique"] = float(len(unique_smiles_set)) / len(valid_smiles_list)
                statistics[f"{stats_name}/novel"] = float(len(novel_smiles_list)) / len(valid_smiles_list)
                statistics[f"{stats_name}/efficient"] = float(len(efficient_smiles_list)) / self.hparams.num_samples_ood
                properties_estimated_unscaled = None
                if self.train_dataset.scaler_properties is not None:
                    if prop_pred is not None:
                        properties_estimated_unscaled = torch.tensor(self.train_dataset.scaler_properties.inverse_transform(prop_pred[idx_valid].cpu().numpy())).to(device=self.device, dtype=properties.dtype)
                    properties_unscaled = torch.tensor(self.train_dataset.scaler_properties.inverse_transform(properties[idx_valid].cpu().numpy())).to(device=self.device, dtype=properties.dtype)
                else:
                    properties_unscaled = properties[idx_valid]
                    if prop_pred is not None:
                        properties_estimated_unscaled = prop_pred[idx_valid]
                        properties_estimated_unscaled = properties_estimated_unscaled[:, idx]
                print(properties_unscaled[0])
                statistics[f"{stats_name}/Min_MAE"], statistics[f"{stats_name}/Min10_MAE"], statistics[f"{stats_name}/Min100_MAE"], statistics[f"{stats_name}/prop_pred_diff1"], statistics[f"{stats_name}/prop_pred_diff10"], statistics[f"{stats_name}/prop_pred_diff100"] = MAE_properties(valid_mols_list, properties=properties_unscaled[:, idx].unsqueeze(1), name=feat_name, properties_estimated=properties_estimated_unscaled, max_value=self.hparams.ood_values[i_]) # molwt, LogP, QED
                print(statistics[f"{stats_name}/valid"])
                print(statistics[f"{stats_name}/Min_MAE"])
                print(statistics[f"{stats_name}/Min10_MAE"])
                print(statistics[f"{stats_name}/Min100_MAE"])
                print(statistics[f"{stats_name}/prop_pred_diff1"])
                print(statistics[f"{stats_name}/prop_pred_diff10"])
                print(statistics[f"{stats_name}/prop_pred_diff100"])
                for key, val in statistics.items():
                    self.log(key, val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)
            
    # Sample from specific range of values for each properties and evaluate using our own estimator
    def check_samples_specific(self):
        assert self.hparams.num_samples % self.hparams.n_gpu == 0
        num_samples = self.hparams.num_samples // self.hparams.n_gpu if not self.trainer.sanity_checking else 2
        stats_name = f"sample_specific"
        specific_max = np.expand_dims(np.array(self.hparams.specific_max, dtype=float), axis=0)
        specific_min = np.expand_dims(np.array(self.hparams.specific_min, dtype=float), axis=0)
        properties_np = np.random.rand(num_samples, self.train_dataset.n_properties)*(specific_max-specific_min) + specific_min
        mask_cond = np.equal(self.hparams.specific_min, 666)
        if len(self.hparams.specific_minmax) == 0:
            props_must_min = np.zeros((len(mask_cond)), dtype=bool)
        else:
            props_must_min = np.equal(self.hparams.specific_minmax, -1)
        if len(self.hparams.specific_minmax) == 0:
            props_must_max = np.zeros((len(mask_cond)), dtype=bool)
        else:
            props_must_max = np.equal(self.hparams.specific_minmax, 1)

        if self.global_rank==0:
            print(f'raw_prop: {properties_np[0]}')
        if self.train_dataset.scaler_properties is not None:
            properties_np = self.train_dataset.scaler_properties.transform(properties_np) # raw to whatever
        if self.global_rank==0:
            print(f'std_prop: {properties_np[0]}')
            print(f'Masked property: {mask_cond}')
            print(stats_name)
        local_properties = torch.tensor(properties_np).to(device=self.device, dtype=torch.float32)
        local_smiles_list, results, local_prop_pred = self.sample_cond(local_properties, num_samples, 
            temperature_min=self.hparams.temperature_min, temperature_max=self.hparams.temperature_max, 
            guidance_min=self.hparams.guidance_min, guidance_max=self.hparams.guidance_max, mask_cond=mask_cond)

        # We will mask all properties except the ones we care about for the MAE
        if len(self.hparams.specific_mae_props) > 0:
            for idx in range(len(mask_cond)):
                mask_cond[idx] = True
            for feat_name in self.hparams.specific_mae_props:
                idx = np.where(feat_name == self.train_dataset.properties_names)[0][0]
                mask_cond[idx] = False
            mask_cond = mask_cond | np.equal(self.hparams.specific_min, 666)

        print("constraints")
        print(len(local_smiles_list))
        idx_valid = []
        valid_smiles_list = []
        for smiles in local_smiles_list:
            if smiles is not None:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None and mol.GetNumHeavyAtoms() <= self.hparams.max_number_of_atoms and max_ring_size(mol) <= self.hparams.max_ring_size:
                    idx_valid += [True]
                    valid_smiles_list += [smiles]
                else:
                    idx_valid += [False]
            else:
                idx_valid += [False]
        local_smiles_list = valid_smiles_list
        local_properties = local_properties[idx_valid]
        local_prop_pred = local_prop_pred[idx_valid]
        print(len(valid_smiles_list))

        if self.train_dataset.scaler_properties is not None: # make into raw
            local_properties = torch.tensor(self.train_dataset.scaler_properties.inverse_transform(local_properties.cpu().numpy())).to(device=self.device, dtype=local_properties.dtype)
            local_prop_pred = torch.tensor(self.train_dataset.scaler_properties.inverse_transform(local_prop_pred.cpu().numpy())).to(device=self.device, dtype=local_properties.dtype)

        for prop_bigger_than, prop_smaller_than, prop_name in zip(self.hparams.props_bigger_than, self.hparams.props_smaller_than, self.hparams.props_names):
            assert prop_name in self.train_dataset.properties_names
            idx = np.where(prop_name == self.train_dataset.properties_names)[0][0]
            print("big-less-than")
            print(len(local_smiles_list))
            idx_valid = []
            valid_smiles_list = []
            for i, smiles in enumerate(local_smiles_list):
                if local_prop_pred[i, idx] <= prop_smaller_than and local_prop_pred[i, idx] >= prop_bigger_than:
                    idx_valid += [True]
                    valid_smiles_list += [smiles]
                else:
                    idx_valid += [False]
            local_smiles_list = valid_smiles_list
            local_properties = local_properties[idx_valid]
            local_prop_pred = local_prop_pred[idx_valid]
            print(len(valid_smiles_list))

        if self.hparams.remove_duplicates:
            print('Removing duplicates')
            print(len(local_smiles_list))
            non_dup_smiles = remove_duplicates(generated_smiles=local_smiles_list, train_smiles=self.train_dataset.smiles_list)
            local_smiles_list = (np.array(local_smiles_list, dtype=object)[non_dup_smiles]).tolist()
            local_properties = local_properties[non_dup_smiles]
            local_prop_pred = local_prop_pred[non_dup_smiles]
            print(len(local_smiles_list))

        if self.hparams.only_get_top_k > 0:
            print('Only keeping top-k')
            indexes_top_k = return_top_k(properties=local_properties[:, ~mask_cond], properties_estimated=local_prop_pred[:, ~mask_cond], mask_cond=None, k=self.hparams.only_get_top_k // self.hparams.n_gpu, props_must_min=props_must_min[~mask_cond], props_must_max=props_must_max[~mask_cond]) 
            local_smiles_list = (np.array(local_smiles_list, dtype=object)[indexes_top_k.cpu()]).tolist()
            local_properties = local_properties[indexes_top_k]
            local_prop_pred = local_prop_pred[indexes_top_k]
            print(len(local_smiles_list))

        oracle_samples = len(local_smiles_list)

        if 'xtb' in self.hparams.dataset_name or self.hparams.xtb_max_energy < 9999 or self.hparams.xtb_min_energy > -9999:
            print('getting XTB predictions')
            current_datetime = str(datetime.datetime.now()).replace(" ", "").replace(".", "::")
            print(current_datetime)
            local_xtb_prop_pred = get_xtb_scores(local_smiles_list, name=self.hparams.tag+current_datetime+str(self.global_rank)) # overwrite prop-preds with the xtb predictions
            local_xtb_prop_pred = torch.tensor(local_xtb_prop_pred).to(device=self.device, dtype=torch.float32)
            if 'wavelength_energy' in self.train_dataset.properties_names:
                idx_wv = np.where('wavelength_energy' == self.train_dataset.properties_names)[0][0]
                local_prop_pred[:, idx_wv] = local_xtb_prop_pred[:, 0]
            if 'f_osc' in self.train_dataset.properties_names:
                idx_osc = np.where('f_osc' == self.train_dataset.properties_names)[0][0]
                local_prop_pred[:, idx_osc] = local_xtb_prop_pred[:, 1]
            # Check and remove any inf/nans
            good_ones = local_prop_pred.isfinite().all(axis=1).tolist()
            print(f"{sum(good_ones)}/{len(good_ones)} were kept after removing Inf or NaNs")
            local_xtb_prop_pred = local_xtb_prop_pred[good_ones]
            local_prop_pred = local_prop_pred[good_ones]
            local_properties = local_properties[good_ones]
            local_smiles_list = (np.array(local_smiles_list, dtype=object)[good_ones]).tolist()

            if self.hparams.xtb_max_energy < 9999 or self.hparams.xtb_min_energy > -9999:
                print(len(local_smiles_list))
                idx_valid = []
                valid_smiles_list = []
                for i, smiles in enumerate(local_smiles_list):
                    if local_xtb_prop_pred[i, 0] <= self.hparams.xtb_max_energy and local_xtb_prop_pred[i, 0] >= self.hparams.xtb_min_energy:
                        idx_valid += [True]
                        valid_smiles_list += [smiles]
                    else:
                        idx_valid += [False]
                local_smiles_list = valid_smiles_list
                local_properties = local_properties[idx_valid]
                local_prop_pred = local_prop_pred[idx_valid]
                print(len(valid_smiles_list))

        # Gather results
        if self.hparams.n_gpu > 1:
            global_smiles_list = [None for _ in range(self.hparams.n_gpu)]
            dist.all_gather_object(global_smiles_list, local_smiles_list)
            smiles_list = []
            for i in range(self.hparams.n_gpu):
                smiles_list += global_smiles_list[i]

            global_properties = [torch.zeros_like(local_properties) for _ in range(self.hparams.n_gpu)]
            dist.all_gather(global_properties, local_properties)
            properties = torch.cat(global_properties, dim=0)

            global_prop_pred = [torch.zeros_like(local_prop_pred) for _ in range(self.hparams.n_gpu)]
            dist.all_gather(global_prop_pred, local_prop_pred)
            prop_pred = torch.cat(global_prop_pred, dim=0)
        else:
            smiles_list = local_smiles_list
            properties = local_properties
            prop_pred = local_prop_pred

        if self.hparams.append_generated_mols_to_file != '' and self.global_rank == 0:
            assert 'xtb' in self.hparams.dataset_name # only with XTB for now
            print(f'Appending generated molecules to file: {self.hparams.append_generated_mols_to_file}')
            properties_generated = prop_pred[:, 0:2].cpu().numpy()
            properties_generated = np.concatenate((np.expand_dims(np.array(smiles_list, dtype=object), axis=1), properties_generated), axis=1)
            properties_generated = properties_generated[0:self.hparams.top_k_to_add] # only keep Top-K
            if os.path.exists(self.hparams.append_generated_mols_to_file):
                prev_props = np.load(self.hparams.append_generated_mols_to_file, allow_pickle=True)
                properties_generated = np.concatenate((prev_props, properties_generated), axis=0)
            with open(self.hparams.append_generated_mols_to_file, 'wb') as f:
                np.save(f, properties_generated)
            print('Done')

        mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        unique_smiles_set = set(smiles_list)
        novel_smiles_list = [smiles for smiles in smiles_list if smiles not in self.train_smiles_set]
        efficient_smiles_list = [smiles for smiles in unique_smiles_set if smiles in novel_smiles_list]
        statistics = dict()

        statistics[f"{stats_name}/valid"] = float(len(smiles_list)) / self.hparams.num_samples
        statistics[f"{stats_name}/unique"] = float(len(unique_smiles_set)) / len(smiles_list)
        statistics[f"{stats_name}/novel"] = float(len(novel_smiles_list)) / len(smiles_list)
        statistics[f"{stats_name}/efficient"] = float(len(efficient_smiles_list)) / self.hparams.num_samples
        if self.global_rank==0:
            print(properties[0])
        statistics[f"{stats_name}/Min_MAE"], statistics[f"{stats_name}/Min10_MAE"], statistics[f"{stats_name}/Min100_MAE"], statistics[f"{stats_name}/Min_sim"], statistics[f"{stats_name}/Min10_sim"], statistics[f"{stats_name}/Min100_sim"], statistics[f"{stats_name}/Min10_div"] = MAE_properties_estimated(mol_list, train_smiles=self.train_dataset.smiles_list, properties=properties[:, ~mask_cond], properties_estimated=prop_pred[:, ~mask_cond], mask_cond=None, zero_rank=self.global_rank==0, properties_all=prop_pred, properties_names=self.train_dataset.properties_names, props_must_min=props_must_min[~mask_cond], props_must_max=props_must_max[~mask_cond], top_k=self.hparams.top_k_output, store_output=self.hparams.specific_store_output, oracle_samples=oracle_samples) 
        if self.global_rank==0:
            print(statistics[f"{stats_name}/valid"])
            print(statistics[f"{stats_name}/Min_MAE"])
            print(statistics[f"{stats_name}/Min10_MAE"])
            print(statistics[f"{stats_name}/Min100_MAE"])
            print(statistics[f"{stats_name}/Min_sim"])
            print(statistics[f"{stats_name}/Min10_sim"])
            print(statistics[f"{stats_name}/Min10_div"])
        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)

    # Sample conditional on a property
    def sample_cond(self, properties, num_samples, guidance_min, guidance_max, temperature_min, temperature_max, mask_cond=None):
        print("Sample_cond")
        offset = 0
        results = []
        prop_pred = []
        self.model.eval()
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            batched_cond_data = properties[offset:(offset+cur_num_samples), :]
            offset += cur_num_samples
            print(offset)
            data_list, prop_pred_ = self.model.decode(batched_cond_data, max_len=self.hparams.max_len, device=self.device, mask_cond=mask_cond,
                temperature_min=temperature_min, temperature_max=temperature_max, 
                guidance_min=guidance_min, guidance_max=guidance_max,
                top_k=self.hparams.top_k, best_out_of_k=self.hparams.best_out_of_k, ranking_based=self.hparams.ranking_based_best_of_k,
                predict_prop=self.hparams.lambda_predict_prop > 0, return_loss_prop=self.hparams.lambda_predict_prop > 0,
                allow_empty_bond=not self.hparams.not_allow_empty_bond, banned_tokens=self.hparams.banned_tokens)
            if prop_pred_ is not None:
                prop_pred += [prop_pred_]
            results.extend((data.to_smiles(), "".join(data.tokens), data.error) for data in data_list)
        if prop_pred_ is not None:
            prop_pred = torch.cat(prop_pred, dim=0)
        if len(prop_pred) == 0:
            prop_pred = None
        self.model.train()
        disable_rdkit_log()
        smiles_list = [canonicalize(elem[0]) for elem in results]
        enable_rdkit_log()

        return smiles_list, results, prop_pred

    # Generate molecules conditional on random properties from the test dataset
    def sample(self, num_samples):
        print("Sample")
        my_loader = DataLoader(
            self.test_dataset,
            batch_size=self.hparams.sample_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers-1,
            drop_last=False,
            persistent_workers=False, pin_memory=True,
        )
        train_loader_iter = iter(my_loader)

        offset = 0
        results = []
        prop_pred = []
        properties = None
        self.model.eval()
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples
            print(offset)
            try:
                _, batched_cond_data = next(train_loader_iter)
            except:
                train_loader_iter = iter(my_loader)
                _, batched_cond_data = next(train_loader_iter)
            while batched_cond_data.shape[0] < cur_num_samples:
                try:
                    _, batched_cond_data_ = next(train_loader_iter)
                except:
                    train_loader_iter = iter(my_loader)
                    _, batched_cond_data_ = next(train_loader_iter)
                batched_cond_data = torch.cat((batched_cond_data, batched_cond_data_), dim=0)
            batched_cond_data = batched_cond_data[:cur_num_samples,:].to(device=self.device)
            if properties is None:
                properties = batched_cond_data
            else:
                properties = torch.cat((properties, batched_cond_data), dim=0)
            data_list, prop_pred_ = self.model.decode(batched_cond_data, max_len=self.hparams.max_len, device=self.device, 
                temperature_min=self.hparams.temperature_min, temperature_max=self.hparams.temperature_max, 
                guidance_min=self.hparams.guidance_min, guidance_max=self.hparams.guidance_max,
                top_k=self.hparams.top_k, best_out_of_k=self.hparams.best_out_of_k, ranking_based=self.hparams.ranking_based_best_of_k,
                predict_prop=self.hparams.lambda_predict_prop > 0,
                allow_empty_bond=not self.hparams.not_allow_empty_bond, banned_tokens=self.hparams.banned_tokens)
            if prop_pred_ is not None:
                prop_pred += [prop_pred_]
            results.extend((data.to_smiles(), "".join(data.tokens), data.error) for data in data_list)
        if prop_pred_ is not None:
            prop_pred = torch.cat(prop_pred, dim=0)
        if len(prop_pred) == 0:
            prop_pred = None
        self.model.train()
        disable_rdkit_log()
        smiles_list = [canonicalize(elem[0]) for elem in results]
        enable_rdkit_log()

        return smiles_list, results, properties, prop_pred

    def check_samples(self):
        assert self.hparams.num_samples % self.hparams.n_gpu == 0
        num_samples = self.hparams.num_samples // self.hparams.n_gpu if not self.trainer.sanity_checking else 2
        local_smiles_list, results, local_properties, _ = self.sample(num_samples)

        if self.hparams.n_gpu > 1:
            # Gather results
            global_smiles_list = [None for _ in range(self.hparams.n_gpu)]
            dist.all_gather_object(global_smiles_list, local_smiles_list)
            smiles_list = []
            for i in range(self.hparams.n_gpu):
                print(i)
                print(len(global_smiles_list[i]))
                smiles_list += global_smiles_list[i]

            global_properties = [torch.zeros_like(local_properties) for _ in range(self.hparams.n_gpu)]
            dist.all_gather(global_properties, local_properties)
            properties = torch.cat(global_properties, dim=0)
        else:
            smiles_list = local_smiles_list
            properties = local_properties

        print('metrics')
        #
        idx_valid = []
        valid_smiles_list = []
        valid_mols_list = []
        for smiles in smiles_list:
            if smiles is not None:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None and mol.GetNumHeavyAtoms() <= self.hparams.max_number_of_atoms and max_ring_size(mol) <= self.hparams.max_ring_size:
                    idx_valid += [True]
                    valid_smiles_list += [smiles]
                    valid_mols_list += [mol]
                else:
                    idx_valid += [False]
            else:
                idx_valid += [False]
                
        unique_smiles_set = set(valid_smiles_list)
        novel_smiles_list = [smiles for smiles in valid_smiles_list if smiles not in self.train_smiles_set]
        efficient_smiles_list = [smiles for smiles in unique_smiles_set if smiles in novel_smiles_list]
        statistics = dict()
        statistics["sample/valid"] = float(len(valid_smiles_list)) / self.hparams.num_samples
        statistics["sample/unique"] = float(len(unique_smiles_set)) / len(valid_smiles_list)
        statistics["sample/novel"] = float(len(novel_smiles_list)) / len(valid_smiles_list)
        statistics["sample/efficient"] = float(len(efficient_smiles_list)) / self.hparams.num_samples
        if self.train_dataset.scaler_properties is not None:
            properties_unscaled = torch.tensor(self.train_dataset.scaler_properties.inverse_transform(properties[idx_valid].cpu().numpy())).to(dtype=torch.float32, device=self.device)
        else:
            properties_unscaled = properties[idx_valid]
        statistics["sample/Min_MAE"], statistics[f"sample/Min10_MAE"], statistics[f"sample/Min100_MAE"], _, _, _ = MAE_properties(valid_mols_list, properties=properties_unscaled) # molwt, LogP, QED
        print(statistics["sample/Min_MAE"])
        print(statistics["sample/Min10_MAE"])
        print(statistics["sample/Min100_MAE"])

        #
        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)
        
        if len(valid_smiles_list) > 0:
            torch.backends.cudnn.enabled = False
            print('get-all-metrics')
            moses_statistics = moses.get_all_metrics(
                smiles_list, 
                n_jobs=self.hparams.num_workers-1,
                device=str(self.device), 
                train=self.train_dataset.smiles_list, 
                test=self.test_dataset.smiles_list,
            )
            print('get-all-metrics done')
            for key in moses_statistics:
                self.log(f"sample/moses/{key}", moses_statistics[key], on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)#, rank_zero_only=True)
            torch.backends.cudnn.enabled = True

    # Generate molecules unconditional
    def sample_uncond(self, num_samples):
        print("Sample Uncond")
        offset = 0
        results = []
        prop_pred = []
        self.model.eval()
        batched_cond_data = torch.zeros(self.hparams.sample_batch_size, 3, device=self.device)
        mask_cond = [True for i in range(self.train_dataset.n_properties)] # mask everything
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples
            print(offset)
            data_list, prop_pred_ = self.model.decode(batched_cond_data[:cur_num_samples], mask_cond=mask_cond, max_len=self.hparams.max_len, device=self.device, 
                temperature_min=self.hparams.temperature_min, temperature_max=self.hparams.temperature_max, 
                guidance_min=self.hparams.guidance_min, guidance_max=self.hparams.guidance_max,
                top_k=self.hparams.top_k, best_out_of_k=self.hparams.best_out_of_k, ranking_based=self.hparams.ranking_based_best_of_k,
                predict_prop=self.hparams.lambda_predict_prop > 0,
                allow_empty_bond=not self.hparams.not_allow_empty_bond, banned_tokens=self.hparams.banned_tokens)
            if prop_pred_ is not None:
                prop_pred += [prop_pred_]
            results.extend((data.to_smiles(), "".join(data.tokens), data.error) for data in data_list)
        if prop_pred_ is not None:
            prop_pred = torch.cat(prop_pred, dim=0)
        if len(prop_pred) == 0:
            prop_pred = None
        self.model.train()
        print("Sample Uncond -done-")
        disable_rdkit_log()
        smiles_list = [canonicalize(elem[0]) for elem in results]
        enable_rdkit_log()

        return smiles_list, results, prop_pred

    def check_samples_uncond(self):
        assert self.hparams.num_samples % self.hparams.n_gpu == 0
        num_samples = self.hparams.num_samples // self.hparams.n_gpu if not self.trainer.sanity_checking else 2
        local_smiles_list, results, _ = self.sample_uncond(num_samples)

        if self.hparams.n_gpu > 1:
            # Gather results
            global_smiles_list = [None for _ in range(self.hparams.n_gpu)]
            dist.all_gather_object(global_smiles_list, local_smiles_list)
            smiles_list = []
            for i in range(self.hparams.n_gpu):
                print(i)
                print(len(global_smiles_list[i]))
                smiles_list += global_smiles_list[i]
        else:
            smiles_list = local_smiles_list

        print('metrics')
        #
        idx_valid = [smiles is not None for smiles in smiles_list]
        valid_smiles_list = [smiles for i, smiles in enumerate(smiles_list) if idx_valid]
        unique_smiles_set = set(valid_smiles_list)
        novel_smiles_list = [smiles for smiles in valid_smiles_list if smiles not in self.train_smiles_set]
        efficient_smiles_list = [smiles for smiles in unique_smiles_set if smiles in novel_smiles_list]
        statistics = dict()
        statistics["sample_uncond/valid"] = float(len(valid_smiles_list)) / self.hparams.num_samples
        statistics["sample_uncond/unique"] = float(len(unique_smiles_set)) / len(valid_smiles_list)
        statistics["sample_uncond/novel"] = float(len(novel_smiles_list)) / len(valid_smiles_list)
        statistics["sample_uncond/efficient"] = float(len(efficient_smiles_list)) / self.hparams.num_samples

        #
        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)
        
        if len(valid_smiles_list) > 0:
            torch.backends.cudnn.enabled = False
            print('get-all-metrics uncond')
            moses_statistics = moses.get_all_metrics(
                smiles_list, 
                n_jobs=self.hparams.num_workers-1,
                device=str(self.device), 
                train=self.train_dataset.smiles_list, 
                test=self.test_dataset.smiles_list,
            )
            print('get-all-metrics uncond done')
            for key in moses_statistics:
                self.log(f"sample_uncond/moses/{key}", moses_statistics[key], on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)#, rank_zero_only=True)
            torch.backends.cudnn.enabled = True

    def check_samples_gflownet(self):
        assert self.hparams.num_samples_gflownet % self.hparams.n_gpu == 0
        num_samples = self.hparams.num_samples_gflownet // self.hparams.n_gpu if not self.trainer.sanity_checking else 2
        stats_name = f"sample"

        properties_np = np.array([self.hparams.gflownet_values]) # desired value to get reward zero
        print(properties_np)
        if self.train_dataset.scaler_properties is not None:
            properties_np = self.train_dataset.scaler_properties.transform(properties_np) # raw to whatever
            print(properties_np)
        properties_np = np.repeat(properties_np, num_samples, axis=0)
        local_properties = torch.tensor(properties_np).to(device=self.device, dtype=torch.float32)
        local_smiles_list, results, _ = self.sample_cond(local_properties, num_samples, 
            temperature_min=self.hparams.temperature_min, temperature_max=self.hparams.temperature_max, 
            guidance_min=self.hparams.guidance_min, guidance_max=self.hparams.guidance_max, mask_cond=None)
        #print('smiles')
        #print(local_smiles_list[0:5])

        # Gather results
        if self.hparams.n_gpu > 1:
            global_smiles_list = [None for _ in range(self.hparams.n_gpu)]
            dist.all_gather_object(global_smiles_list, local_smiles_list)
            smiles_list = []
            for i in range(self.hparams.n_gpu):
                smiles_list += global_smiles_list[i]

            global_properties = [torch.zeros_like(local_properties) for _ in range(self.hparams.n_gpu)]
            dist.all_gather(global_properties, local_properties)
            properties = torch.cat(global_properties, dim=0)
        else:
            smiles_list = local_smiles_list
            properties = local_properties

        idx_valid = []
        valid_smiles_list = []
        valid_mols_list = []
        for smiles in smiles_list:
            if smiles is not None:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None and mol.GetNumHeavyAtoms() <= self.hparams.max_number_of_atoms and max_ring_size(mol) <= self.hparams.max_ring_size:
                    idx_valid += [True]
                    valid_smiles_list += [smiles]
                    valid_mols_list += [mol]
                else:
                    idx_valid += [False]
            else:
                idx_valid += [False]
        unique_smiles_set = set(valid_smiles_list)
        novel_smiles_list = [smiles for smiles in valid_smiles_list if smiles not in self.train_smiles_set]
        efficient_smiles_list = [smiles for smiles in unique_smiles_set if smiles in novel_smiles_list]
        statistics = dict()

        statistics[f"sample/valid"] = float(len(valid_smiles_list)) / self.hparams.num_samples_gflownet
        statistics[f"sample/unique"] = float(len(unique_smiles_set)) / len(valid_smiles_list)
        statistics[f"sample/novel"] = float(len(novel_smiles_list)) / len(valid_smiles_list)
        statistics[f"sample/efficient"] = float(len(efficient_smiles_list)) / self.hparams.num_samples_gflownet
        if self.train_dataset.scaler_properties is not None:
            properties_unscaled = torch.tensor(self.train_dataset.scaler_properties.inverse_transform(properties[idx_valid].cpu().numpy())).to(device=self.device, dtype=properties.dtype)
        else:
            properties_unscaled = properties[idx_valid]
        print(properties_unscaled[0])

        statistics[f"gflownet/weighted_reward"], statistics[f"gflownet/weighted_diversity"], statistics[f"gflownet/mean_reward"], statistics[f"gflownet/mean_diversity"], statistics[f"gflownet/reward0"], statistics[f"gflownet/reward1"], statistics[f"gflownet/reward2"], statistics[f"gflownet/reward3"], top_10_smiles = best_rewards_gflownet(valid_smiles_list, valid_mols_list, device=self.device)
        print("Top-10 molecules")
        print(top_10_smiles)
        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)

    # Property-conditional on test properties with specific metrics
    def check_samples_dit(self):

        print("Random forest classifier for the non-rdkit property")
        model_path = f"../resource/data/{self.hparams.dataset_name}/forest_model.csv.gz"
        forest_model = TaskModel(model_path, self.train_dataset.smiles_list, self.train_dataset.properties, 
            smiles_list_valid=self.test_dataset.smiles_list, properties_valid=self.test_dataset.properties, 
            i = 0, task_type = 'classification')

        print('Intermediate statistics for Frechet distance on test set')
        stats_path = f"../resource/data/{self.hparams.dataset_name}/fcd_stats.npy"
        try:
            stat_ref = load(stats_path)
        except:
            torch.backends.cudnn.enabled = False
            stat_ref = compute_intermediate_statistics(model_path, self.test_dataset.smiles_list, n_jobs=self.hparams.num_workers-1, device=self.device, batch_size=512)
            torch.backends.cudnn.enabled = True

        num_samples = self.hparams.num_samples // self.hparams.n_gpu if not self.trainer.sanity_checking else 2
        if self.hparams.test_on_train_data:
            smiles_list = self.train_dataset.smiles_list
            properties = self.train_dataset.properties
        elif self.hparams.test_on_test_data:
            smiles_list = self.test_dataset.smiles_list
            properties = self.test_dataset.properties
        else:
            local_smiles_list, results, local_properties, _ = self.sample(num_samples)
            if self.hparams.n_gpu > 1:
                # Gather results
                global_smiles_list = [None for _ in range(self.hparams.n_gpu)]
                dist.all_gather_object(global_smiles_list, local_smiles_list)
                smiles_list = []
                for i in range(self.hparams.n_gpu):
                    smiles_list += global_smiles_list[i]

                global_properties = [torch.zeros_like(local_properties) for _ in range(self.hparams.n_gpu)]
                dist.all_gather(global_properties, local_properties)
                properties = torch.cat(global_properties, dim=0)
            else:
                smiles_list = local_smiles_list
                properties = local_properties

        print('metrics')
        idx_valid = [smiles is not None and Chem.MolFromSmiles(smiles) is not None for smiles in smiles_list]
        valid_smiles_list = [smiles for i, smiles in enumerate(smiles_list) if idx_valid]
        unique_smiles_set = set(valid_smiles_list)
        novel_smiles_list = [smiles for smiles in valid_smiles_list if smiles not in self.train_smiles_set]
        efficient_smiles_list = [smiles for smiles in unique_smiles_set if smiles in novel_smiles_list]
        statistics = dict()
        statistics["sample/valid"] = float(len(valid_smiles_list)) / self.hparams.num_samples
        statistics["sample/unique"] = float(len(unique_smiles_set)) / len(valid_smiles_list)
        statistics["sample/novel"] = float(len(novel_smiles_list)) / len(valid_smiles_list)
        statistics["sample/efficient"] = float(len(efficient_smiles_list)) / self.hparams.num_samples
        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)

        if self.train_dataset.scaler_properties is not None:
            if self.hparams.test_on_train_data or self.hparams.test_on_test_data:
                properties_unscaled = self.train_dataset.scaler_properties.inverse_transform(properties)
            else:
                properties_unscaled = self.train_dataset.scaler_properties.inverse_transform(properties.cpu().numpy())
        else:
            if self.hparams.test_on_train_data or self.hparams.test_on_test_data:
                properties_unscaled = properties
            else:
                properties_unscaled = properties.cpu().numpy()
        torch.backends.cudnn.enabled = False
        metrics = compute_molecular_metrics(task_name=self.hparams.dataset_name, molecule_list=smiles_list, targets=properties_unscaled, stat_ref=stat_ref, task_evaluator=forest_model, n_jobs=self.hparams.num_workers-1, device=self.device, batch_size=512)
        torch.backends.cudnn.enabled = True
        for key, val in metrics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)

    def check_sim(self):
        gen_smiles = [smile for smile in self.hparams.check_sim]
        statistics = dict()
        statistics["sample/similarity"], smiles_closests, smiles_closests2, smiles_closests3 = average_similarity2(gen_smiles, self.train_dataset.smiles_list, device=self.device)
        print(self.hparams.check_sim)
        print(statistics["sample/similarity"])
        print(smiles_closests)
        print(smiles_closests2)
        print(smiles_closests3)
        #for key, val in statistics.items():
        #    self.log(key, val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)

    @staticmethod
    def add_args(parser):
        #
        parser.add_argument("--dataset_name", type=str, default="zinc") # zinc, qm9, moses, chromophore, hiv, bbbp, bace

        parser.add_argument("--extra_dataset_name", type=str, default="") # when not empty, the dataset is added as extra
        parser.add_argument("--pretrain", action="store_true") # if true, we are pre-training, so we combined the datasets scaler and vocabs, but not the molecules and properties
        parser.add_argument("--finetune", action="store_true") # if true, we are fine-tuning, so we combined the datasets scaler and vocabs, but not the molecules and properties
        parser.add_argument("--scaler_vocab", type=str, default="combined") # combined [original+extra], original, base (which dataset to use for the scaler when using an extra dataset, this may be important since one dataset may be completely OOD to the other dataset)

        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=6)
        parser.add_argument("--randomize_order", action="store_true") # randomize order of nodes and edges for the spanning tree to produce more diversity
        parser.add_argument("--scaling_type", type=str, default="std") # scaling used on properties (none, std, quantile, minmax)
        parser.add_argument("--start_random", action="store_true") # We can already randomize the order, but this also make the starting atom random

        #
        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--emb_size", type=int, default=1024)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--input_dropout", type=float, default=0.0)
        parser.add_argument("--dropout", type=float, default=0.1) # 0.0 for llms
        parser.add_argument("--logit_hidden_dim", type=int, default=256)
        parser.add_argument("--lambda_predict_prop", type=float, default=0.0) # EOS is used to get an output embedding which is put into a new prediction head to predict all properties of the molecule
        parser.add_argument("--lambda_predict_prop_always", action="store_true") # must predict the properties at every output, not just the EOS one
        parser.add_argument("--best_out_of_k", type=int, default=1) # If >1, we sample k molecules and choose the best out of the k based on the unconditional model of the generated mol property-prediction (IF THERE IS NO PROP-PREDICTION, we just take the first valid ones).
        parser.add_argument("--ranking_based_best_of_k", action="store_true") # If True, uses ranking_based selection (this gives us highest quality molecules that are not duplicated; this only make sense to use when we condition on a single set of properties; do not use this when sampling from multiple different properties)

        parser.add_argument("--gpt", action="store_true") # use a better Transformer with Flash-attention
        parser.add_argument("--no_bias", action="store_true") # bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        parser.add_argument("--rotary", action="store_true") # rotary embedding
        parser.add_argument("--rmsnorm", action="store_true") # RMSNorm instead of LayerNorm
        parser.add_argument("--swiglu", action="store_true") # SwiGLU instead of GELU
        parser.add_argument("--expand_scale", type=float, default=2.0) # expand factor for the MLP
        parser.add_argument("--special_init", action="store_true") # the init used in GPT-2, slows down training a bit though

        parser.add_argument("--cond_lin", action="store_true") # like STGG, single linear layer for continuous variables
        parser.add_argument("--bin_loss_type", type=str, default='ce') # ce, L2, hinge loss for binary properties
        #
        parser.add_argument("--disable_treeloc", action="store_true")
        parser.add_argument("--disable_graphmask", action="store_true")
        parser.add_argument("--disable_valencemask", action="store_true")
        parser.add_argument("--enable_absloc", action="store_true")
        parser.add_argument("--disable_counting_ring", action="store_true") # new
        
        #
        parser.add_argument("--lr", type=float, default=2e-4) # varies for llms
        parser.add_argument("--warmup_steps", type=int, default=0) # 200-1k should be good
        parser.add_argument("--lr_decay", type=float, default=1.0) # 0.1 for llms
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.999) # 0.95 for llms
        parser.add_argument("--weight_decay", type=float, default=0.0) # 0.1 for llms
        parser.add_argument("--disable_random_prop_mask", action="store_true") # new
        parser.add_argument("--not_allow_empty_bond", action="store_true") # use to disable empty bonds 

        #
        parser.add_argument("--max_len", type=int, default=250) # A bit higher to handle OOD
        parser.add_argument("--n_correct", type=int, default=20) # max_len=250 with n_correct=10 means that at len=240 we force the spanning-tree to close itself ASAP to prevent an incomplete spanning-tree
        parser.add_argument("--check_sample_every_n_epoch", type=int, default=10)
        parser.add_argument("--save_every_n_epoch", type=int, default=5) # how often to save checkpoints
        parser.add_argument("--num_samples", type=int, default=10000)
        parser.add_argument("--num_samples_ood", type=int, default=2000)
        parser.add_argument("--sample_batch_size", type=int, default=1250)
        # Manually set OOD: need to be min,max for all properties, so should be 6 values
        # For Zinc based on https://arxiv.org/pdf/2208.10718, values should be 580, 84, 8.194, -3.281, 1.2861, 0.1778
        parser.add_argument("--no_ood", action="store_true") # Do not do OOD sampling
        parser.add_argument("--ood_values", nargs='+', type=float, default=[0.0]) # [min,max,min,max,...] 2*n_properties
        parser.add_argument("--ood_names", nargs='+', type=str, default=['QED']) # names of which properties to sample from n_properties
        parser.add_argument("--only_ood", action="store_true") # Only do OOD sampling

        parser.add_argument("--specific_conditioning", action="store_true") # if True sample with this conditioning value (uses --num_samples)
        parser.add_argument("--specific_min", nargs='+', type=float, default=[]) # If 2 properties, can set 0.5 2 or 0.5 666 (Any 666 is interpreted as missing)
        parser.add_argument("--specific_max", nargs='+', type=float, default=[]) # If 2 properties, can set 0.5 2 or 0.5 666 (Any 666 is interpreted as missing)
        parser.add_argument("--specific_minmax", nargs='+', type=float, default=[]) # For each specific_values, specify if they must be minimized (-1), maximized (1), or run the L2 MAE (0)
        parser.add_argument("--specific_mae_props", nargs='+', type=str, default=[]) # If provided, only does MAE on those properties
        parser.add_argument("--specific_store_output", type=str, default="") # ex:"my/folder/myfile.csv" where to put csv file where to store outputs (if saving is desired)

        # Tunable knobs post-training
        parser.add_argument("--temperature_min", type=float, default=1.0)
        parser.add_argument("--temperature_max", type=float, default=1.0)
        parser.add_argument("--guidance_min", type=float, default=1.0) 
        parser.add_argument("--guidance_max", type=float, default=1.0)
        parser.add_argument("--top_k", type=int, default=0) # if > 0, we only select from the top-k tokens
        #(1-gamma)*model(generated_seq_no_prompt) + gamma*model(generated_seq_with_prompt)

        # Replicating the conditioning of the Gflownet https://arxiv.org/abs/2210.12765 experiments and metrics
        parser.add_argument("--gflownet", action="store_true")
        parser.add_argument("--gflownet_realgap", action="store_true")
        parser.add_argument("--gflownet_values", nargs='+', type=float, default=[0.5, 2.5, 1.0, 105.0])
        parser.add_argument("--num_samples_gflownet", type=int, default=128)

        # Only for DiT sampling for now
        parser.add_argument("--test_on_train_data", action="store_true") # If True, we use --test on the train data instead of fake generated data
        parser.add_argument("--test_on_test_data", action="store_true") # If True, we use --test on the test data instead of fake generated data

        parser.add_argument("--no_test_set_accuracy", action="store_true") # If True, do not compute the global test accuracy to reduce memoru
        parser.add_argument("--no_test_step", action="store_true") # ignore test set
        parser.add_argument("--legacy_sort", action="store_true") # for old models which didn't sort the atom list

        # xtb hyperparameters
        parser.add_argument("--xtb_max_energy", type=float, default=9999.0) # Will make all molecules with energy < x removed (we want wavelength > 1000, so energy < 1.2)
        parser.add_argument("--xtb_min_energy", type=float, default=-9999.0) # Will make all molecules with energy < x removed (we want wavelength > 1000, so energy < 1.2)
        parser.add_argument("--xtb_reward_type", type=str, default="none") # none, f_osc or IR_f_osc

        # properties to use (Default: 3 properties of MolWt, LogP, and QED)
        parser.add_argument("--limited_properties", action="store_true") # If True, don't use extra properties
        parser.add_argument("--force_prop_redo", action="store_true") # If True, force properties recalculation

        parser.add_argument("--mask_prop_in_original", nargs='+', type=str, default=[]) # mask the properties 
        parser.add_argument("--mask_prop_in_extra", nargs='+', type=str, default=[]) # mask the properties

        parser.add_argument("--mask_seperately", action="store_true") # mask extra properties seperately
        parser.add_argument("--extra_props_dont_always_mask", nargs='+', type=str, default=[]) # these properties wont be masked seperately
        parser.add_argument("--remove_properties", nargs='+', type=str, default=[]) # remove those properties from the dataset properties

        parser.add_argument("--max_number_of_atoms", type=int, default=9999) # Will make all molecules with more than k heavy atoms invalid and removed (80-100 is reasonable).
        parser.add_argument("--max_ring_size", type=int, default=9999) # Will make all molecules with rings of more than k heavy atoms invalid and removed (8 is reasonable).

        parser.add_argument("--portion_used", type=float, default=1.0) # portion of the dataset used (always use 1.0 unless you want to test at different number-of-samples)

        # Active Learning (aka add generated samples and their properties back into the finetune dataset to have bigger and better dataset)
        parser.add_argument("--append_generated_mols_to_file", type=str, default='') # only works with XTB for now since its the only implemented high-accuracy property-predictor; append new molecules with XTB properties to a file
        parser.add_argument("--top_k_to_add", type=int, default=9999999) # Instead of adding all generated molecules, it will add the top-K molecules
        parser.add_argument("--load_generated_mols", nargs='+', type=str, default=[]) # load one or multiple files containing molecules and their properties, add it to the dataset (molecules with new tokens are automatically removed)
        parser.add_argument("--load_generated_mols_extra", nargs='+', type=str, default=[]) # load one or multiple files containing molecules and their properties, add it to the extra dataset
        parser.add_argument("--dont_remove_unseen_tokens", action="store_true") # If True, we do not automatically remove new tokens from --load_generated_mols

        parser.add_argument("--banned_tokens", nargs='+', type=str, default=[]) # banned tokens
        # Trim down the list of generated molecules by some constraints after generation
        parser.add_argument("--props_bigger_than", nargs='+', type=float, default=[])
        parser.add_argument("--props_smaller_than", nargs='+', type=float, default=[])
        parser.add_argument("--props_names", nargs='+', type=str, default=[]) # properties names for bigger/smaller-than
        parser.add_argument("--only_get_top_k", type=int, default=0) # If > 0, only keep the top-k molecules generated (this is applied before doing XTB! So it reduces compute demand when using XTB)

        parser.add_argument("--remove_duplicates", action="store_true") # If true, remove generated molecules that exist in the training dataset or the other generated molecules
        parser.add_argument("--top_k_output", type=int, default=100) # How many of the top-k molecules to print

        parser.add_argument("--check_sim", nargs='+', type=str, default=[]) # If non-empty, only compute the similarity to these molecules and nothing else

        # To enforce a maximum simalarity based on set of molecules
        parser.add_argument("--max_similarity", type=float, default=1.0)

        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    CondGeneratorLightningModule.add_args(parser)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--save_checkpoint_dir", type=str, default="CHANGE_TO_YOUR_DIR/AutoregressiveMolecules_checkpoints")
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--val_acc_tracking", action="store_true") # If False, track lowest val loss, if True, track lowest val accuracy
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--last", action="store_true")
    hparams = parser.parse_args()

    if hparams.dataset_name in ['bbbp', 'bace', 'hiv']:
        hparams.graph_dit_benchmark = True
    else:
        hparams.graph_dit_benchmark = False

    print('Warning: Note that for both training and metrics, results will only be reproducible when using the same number of GPUs and num_samples/sample_batch_size')
    pl.seed_everything(hparams.seed, workers=True) # use same seed, except for the dataloaders
    model = CondGeneratorLightningModule(hparams)
    if hparams.load_checkpoint_path != "":
        model.load_state_dict(torch.load(hparams.load_checkpoint_path)["state_dict"])
    if hparams.compile:
        model = torch.compile(model)

    neptune_logger = NeptuneLogger(
        api_key="YOUR_API_KEY",
        project="YOUR_PROJECT_KEY",
        source_files="**/*.py",
        tags=hparams.tag.split("_"),
        log_model_checkpoints=False,
        )
    neptune_logger.log_hyperparams(vars(hparams))
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(hparams.save_checkpoint_dir, hparams.tag), 
        save_last=True,
        every_n_epochs=hparams.save_every_n_epoch,
        enable_version_counter=False,
    )

    if hparams.n_gpu > 1:
        if not hparams.test:
            strategy="ddp_find_unused_parameters_true" 
        else:
            strategy=DDPStrategy(timeout=datetime.timedelta(seconds=7200000))
    else:
        strategy='auto'

    trainer = pl.Trainer(
        devices="auto" if hparams.cpu else hparams.n_gpu, 
        accelerator="cpu" if hparams.cpu else "gpu",
        strategy=strategy,
        precision="bf16-mixed" if hparams.bf16 else "32-true",
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=hparams.gradient_clip_val,
        log_every_n_steps=hparams.log_every_n_steps,
        limit_val_batches=0 if hparams.test or hparams.no_test_step else None,
        num_sanity_val_steps=0 if hparams.test or hparams.no_test_step else 2,
    )
    pl.seed_everything(hparams.seed + trainer.global_rank, workers=True) # different seed per worker
    trainer.fit(model, 
            ckpt_path='last')
    pl.seed_everything(hparams.seed + trainer.global_rank, workers=True) # different seed per worker
    if hparams.test:
        if hparams.last:
            trainer.test(model, 
                ckpt_path='last')
        else:
            trainer.test(model, 
                ckpt_path=checkpoint_callback.best_model_path)
