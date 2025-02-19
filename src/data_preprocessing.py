import os
import argparse
from pathlib import Path
import numpy as np
from props.properties import MolLogP_smiles, qed_smiles, ExactMolWt_smiles
from data.target_data import Data as TargetData
from data.dataset import get_cond_datasets
from evaluate.MCD.evaluator import TaskModel
from rdkit import Chem
DATA_DIR = "../resource/data"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='zinc')
    parser.add_argument("--MAX_LEN", type=int, default=250)
    parser.add_argument("--force_vocab_redo", action="store_true")
    parser.add_argument("--force_prop_redo", action="store_true")
    parser.add_argument("--predictors", action="store_true")
    parser.add_argument("--limited_properties", action="store_true") # If True, don't use extra properties

    params = parser.parse_args()

    raw_dir = f"{DATA_DIR}/{params.dataset_name}"
    datasets = get_cond_datasets(dataset_name=params.dataset_name, raw_dir=raw_dir, randomize_order=False, 
                MAX_LEN=params.MAX_LEN, scaling_type='std', gflownet=False, 
                force_vocab_redo=params.force_vocab_redo, force_prop_redo=params.force_prop_redo,
                limited_properties=params.limited_properties)

    for split, dataset in zip(['train', 'valid', 'test'], datasets):
        max_len = 0
        smile = max(dataset.smiles_list, key = len)
        print("---------------")
        print(smile)
        smile_transformed = TargetData.from_smiles(smile, dataset.vocab, randomize_order=False, MAX_LEN=params.MAX_LEN)
        print(smile_transformed.sequence)
        print("".join(smile_transformed.tokens))
        smile_reversed = smile_transformed.to_smiles()
        print(smile_reversed)
        new_len = len(smile_transformed.sequence)
        max_len = max(max_len, new_len)
        print(f"split={split} Approximate-Max-Length={max_len}")

    if params.predictors:
        print("Random forest predictors")
        for i in range(datasets[0].properties.shape[1]):
            model_path = f"../resource/data/{params.dataset_name}/forest_model_{i}.csv.gz"
            forest_model = TaskModel(model_path, datasets[0].smiles_list, datasets[0].properties, 
                smiles_list_valid=datasets[2].smiles_list, properties_valid=datasets[2].properties, 
                i = i, task_type = 'regression')
