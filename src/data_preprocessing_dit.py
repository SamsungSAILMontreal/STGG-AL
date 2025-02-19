## Needed for the metrics in https://arxiv.org/pdf/2401.13858

import os
import argparse
from pathlib import Path
import numpy as np
import torch
from evaluate.MCD.evaluator import TaskModel, BasicMolecularMetrics
from moses.metrics.metrics import compute_intermediate_statistics
from joblib import dump, load
from data.dataset import get_cond_datasets
from data.target_data import Data as TargetData
from evaluate.MCD.evaluator import TaskModel
DATA_DIR = "../resource/data"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='bace') # bace, bbbp, hiv
    parser.add_argument("--MAX_LEN", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=24)
    hparams = parser.parse_args()

    raw_dir = f"../resource/data/{hparams.dataset_name}"

    datasets = get_cond_datasets(dataset_name=hparams.dataset_name, raw_dir=raw_dir, randomize_order=False, 
                MAX_LEN=hparams.MAX_LEN, scaling_type='std', 
                gflownet=False)
    train_dataset, val_dataset, test_dataset = datasets

    print("Training a random forest classifier for the non-rdkit property")
    model_path = f"../resource/data/{hparams.dataset_name}/forest_model.csv.gz"
    forest_model = TaskModel(model_path, train_dataset.smiles_list, train_dataset.properties, 
        i = 0, task_type = 'classification',
        smiles_list_valid=test_dataset.smiles_list, properties_valid=test_dataset.properties)

    print('Calculating intermediate statistics for Frechet distance on test set')
    stats_path = f"../resource/data/{hparams.dataset_name}/fcd_stats.npy"
    torch.backends.cudnn.enabled = False
    stat_ref = compute_intermediate_statistics(test_dataset.smiles_list, n_jobs=hparams.num_workers, device='cuda', batch_size=512)
    dump(stat_ref, stats_path)

    for split, dataset in zip(['train', 'valid', 'test'], datasets):
        max_len = 0
        smile = max(dataset.smiles_list, key = len)
        print(smile)
        smile_transformed = TargetData.from_smiles(smile, dataset.vocab, randomize_order=False, MAX_LEN=hparams.MAX_LEN)
        print(smile_transformed.sequence)
        smile_transformed.featurize()
        smile_reversed = smile_transformed.to_smiles()
        print(smile_reversed)
        new_len = len(smile_transformed.sequence)
        max_len = max(max_len, new_len)
        print(f"split={split} Approximate-Max-Length={max_len}")

    print("Random forest predictors")
    for i in range(1, datasets[0].properties.shape[1]):
        model_path = f"../resource/data/{hparams.dataset_name}/forest_model_{i}.csv.gz"
        forest_model = TaskModel(model_path, datasets[0].smiles_list, datasets[0].properties, 
            smiles_list_valid=datasets[2].smiles_list, properties_valid=datasets[2].properties, 
            i = i, task_type = 'regression')

    print('Finished')
