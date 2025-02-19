import os
import glob
import copy
from pathlib import Path
import torch
from joblib import dump, load
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset
from data.target_data import Data as TargetData
from data.smiles import smiles_list2atoms_list, get_max_valence_from_dataset
from data.target_data import SpanningTreeVocabulary, merge_vocabs
from props.properties import penalized_logp, MolLogP_smiles, qed_smiles, ExactMolWt_smiles, get_xtb_scores
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from rdkit import Chem
from rdkit.Chem.QED import qed
from rdkit.Chem import Descriptors, rdMolDescriptors, ResonanceMolSupplier, Crippen, GraphDescriptors
from rdkit.Chem.EnumerateStereoisomers import GetStereoisomerCount
import props.sascorer as sascorer
DATA_DIR = "../resource/data"


def max_ring_size(mol):
    max_size_Rings = 0
    ri = mol.GetRingInfo()
    for ring in ri.BondRings():
        max_size_Rings = max(max_size_Rings, len(ring))
    return max_size_Rings

class DummyDataset(Dataset): 
    def __init__(self):
        self.n_properties = 1

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.zeros(1,1,1).to(dtype=torch.float32), torch.zeros(self.n_properties).to(dtype=torch.float32)

def invert_permutation_numpy(permutation):
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv

def load_or_create_3prop(smiles_list, properties_path, force_prop_redo=False):
    redo = not os.path.exists(properties_path) or force_prop_redo
    if not redo:
        properties = np.load(properties_path).astype(float)
        if properties.shape[0] != len(smiles_list):
            redo = True
    if redo:
        print("Properties dataset does not exists, making it")
        molwt = np.expand_dims(np.array(ExactMolWt_smiles(smiles_list, num_workers=6)), axis=1)
        mollogp = np.expand_dims(np.array(MolLogP_smiles(smiles_list, num_workers=6)), axis=1)
        molqed = np.expand_dims(np.array(qed_smiles(smiles_list, num_workers=6)), axis=1)
        properties = np.concatenate((molwt, mollogp, molqed), axis=1) # molwt, LogP, QED
        print(properties_path)
        if os.path.exists(properties_path):
            os.remove(properties_path)
        with open(properties_path, 'wb') as f:
            np.save(f, properties)
        print('Finished making the properties, saving it to file')
        properties = properties.astype(float)
    return properties, np.array(['ExactMolWt','MolLogP','QED'], dtype=object)

class ColumnTransformer(): # Actually working ColumnTransformer, unlike the badly designed sklearn version which change the order the variables (complete nonsense)
    
    def __init__(self, column_transformer, continuous_prop, categorical_prop):
        self.column_transformer = column_transformer
        self.continuous_prop = continuous_prop
        self.categorical_prop = categorical_prop # for now categorical is treated as is since we only have 0/1 variables, but ideally we dummy-code it

    def fit(self, X):
        self.column_transformer.fit(X[:, self.continuous_prop])

    def transform(self, X):
        Y = copy.deepcopy(X)
        Y[:, self.continuous_prop] = self.column_transformer.transform(X[:, self.continuous_prop])
        return Y

    def inverse_transform(self, X):
        Y = copy.deepcopy(X)
        Y[:, self.continuous_prop] = self.column_transformer.inverse_transform(X[:, self.continuous_prop])
        return Y

class ColumnTransformer2(): # Actually working ColumnTransformer, unlike the badly designed sklearn version which change the order the variables (complete nonsense)
    
    def __init__(self, column_transformer0, column_transformer1):
        self.column_transformer0 = column_transformer0
        self.column_transformer1 = column_transformer1

    def fit(self, X1, X2):
        self.column_transformer0.fit(X1[:, 0:1])
        self.column_transformer1.fit(X2[:, 1:2])

    def transform(self, X):
        Y = copy.deepcopy(X)
        Y[:, 0:1] = self.column_transformer0.transform(X[:, 0:1])
        Y[:, 1:2] = self.column_transformer1.transform(X[:, 1:2])
        return Y

    def inverse_transform(self, X):
        Y = copy.deepcopy(X)
        Y[:, 0:1] = self.column_transformer0.inverse_transform(X[:, 0:1])
        Y[:, 1:2] = self.column_transformer1.inverse_transform(X[:, 1:2])
        return Y

class ZincDataset(Dataset):
    raw_dir = f"{DATA_DIR}/zinc"
    def __init__(self, split, randomize_order, MAX_LEN, vocab=None):
        smiles_list_path = os.path.join(self.raw_dir, f"{split}.txt")
        self.smiles_list = Path(smiles_list_path).read_text(encoding="utf=8").splitlines()
        self.vocab = vocab
        self.randomize_order = randomize_order
        self.MAX_LEN = MAX_LEN
        
    def __len__(self):
        return len(self.smiles_list)

    def update_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        return TargetData.from_smiles(smiles, self.vocab, randomize_order=self.randomize_order, MAX_LEN=self.MAX_LEN).featurize()

class QM9Dataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/qm9"

class SimpleMosesDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/moses"

class MosesDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/moses"

class LogPZincDataset(Dataset):
    raw_dir = f"{DATA_DIR}/zinc"
    def __init__(self, split, randomize_order, MAX_LEN, vocab=None):
        smiles_list_path = os.path.join(self.raw_dir, f"{split}.txt")
        self.smiles_list = Path(smiles_list_path).read_text(encoding="utf=8").splitlines()
        self.vocab = vocab
        self.randomize_order = randomize_order
        self.MAX_LEN = MAX_LEN
        
    def update_vocab(self, vocab):
        self.vocab = vocab

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        return TargetData.from_smiles(smiles, self.vocab, randomize_order=self.randomize_order, MAX_LEN=self.MAX_LEN).featurize(), torch.tensor([penalized_logp(smiles)])

def random_data_split(n, data_dir, data, train_ratio = 0.6, valid_ratio = 0.2, test_ratio = 0.2):
    if os.path.exists(os.path.join(data_dir, f'train_idx_{data}_{train_ratio}_{valid_ratio}_{test_ratio}.json')):
        with open(os.path.join(data_dir, f'train_idx_{data}_{train_ratio}_{valid_ratio}_{test_ratio}.json')) as f:
            train_index = json.load(f)
        with open(os.path.join(data_dir, f'val_idx_{data}_{train_ratio}_{valid_ratio}_{test_ratio}.json')) as f:
            val_index = json.load(f)
        with open(os.path.join(data_dir, f'test_idx_{data}_{train_ratio}_{valid_ratio}_{test_ratio}.json')) as f:
            test_index = json.load(f)
        if n != len(train_index) + len(test_index) + len(val_index): # redo, because data must have changed
            full_idx = list(range(n))
            train_index, test_index, _, _ = train_test_split(full_idx, full_idx, test_size=test_ratio, random_state=42)
            train_index, val_index, _, _ = train_test_split(train_index, train_index, test_size=valid_ratio/(valid_ratio+train_ratio), random_state=42)
            with open(os.path.join(data_dir, f'train_idx_{data}_{train_ratio}_{valid_ratio}_{test_ratio}.json'), 'w') as f:
                json.dump(train_index, f)
            with open(os.path.join(data_dir, f'val_idx_{data}_{train_ratio}_{valid_ratio}_{test_ratio}.json'), 'w') as f:
                json.dump(val_index, f)
            with open(os.path.join(data_dir, f'test_idx_{data}_{train_ratio}_{valid_ratio}_{test_ratio}.json'), 'w') as f:
                json.dump(test_index, f)
    else:
        full_idx = list(range(n))
        train_index, test_index, _, _ = train_test_split(full_idx, full_idx, test_size=test_ratio, random_state=42)
        train_index, val_index, _, _ = train_test_split(train_index, train_index, test_size=valid_ratio/(valid_ratio+train_ratio), random_state=42)
        with open(os.path.join(data_dir, f'train_idx_{data}_{train_ratio}_{valid_ratio}_{test_ratio}.json'), 'w') as f:
            json.dump(train_index, f)
        with open(os.path.join(data_dir, f'val_idx_{data}_{train_ratio}_{valid_ratio}_{test_ratio}.json'), 'w') as f:
            json.dump(val_index, f)
        with open(os.path.join(data_dir, f'test_idx_{data}_{train_ratio}_{valid_ratio}_{test_ratio}.json'), 'w') as f:
            json.dump(test_index, f)
    print('dataset len', n, 'train len', len(train_index), 'val len', len(val_index), 'test len', len(test_index))
    return train_index, val_index, test_index

def fixed_data_split(n, data_dir, data):
    full_idx = list(range(n))
    # Test idx
    if data == "qm9":
        if os.path.exists(os.path.join(data_dir, f'test_idx_{data}_.json')):
            with open(os.path.join(data_dir, f'test_idx_{data}_.json')) as f:
                test_index = json.load(f)
        else:
            with open(os.path.join(data_dir, f'test_idx_{data}.json')) as f:
                test_index = json.load(f)
            test_index = test_index['test_idxs']
            test_index = [int(i) for i in test_index]
            with open(os.path.join(data_dir, f'test_idx_{data}_.json'), 'w') as f:
                json.dump(test_index, f)
    else:
        with open(os.path.join(data_dir, f'test_idx_{data}.json')) as f:
            test_index = json.load(f)

    if os.path.exists(os.path.join(data_dir, f'train_idx_{data}.json')):
        with open(os.path.join(data_dir, f'train_idx_{data}.json')) as f:
            train_index = json.load(f)
        with open(os.path.join(data_dir, f'val_idx_{data}.json')) as f:
            val_index = json.load(f)
    else:
        train_index = [i for i in full_idx if i not in test_index]
        train_index, val_index, _, _ = train_test_split(train_index, train_index, test_size=0.05, random_state=42)
        with open(os.path.join(data_dir, f'train_idx_{data}.json'), 'w') as f:
            json.dump(train_index, f)
        with open(os.path.join(data_dir, f'val_idx_{data}.json'), 'w') as f:
            json.dump(val_index, f)
    print('dataset len', n, 'train len', len(train_index), 'val len', len(val_index), 'test len', len(test_index))
    return train_index, val_index, test_index

class PropCondDataset(Dataset): # molwt, LogP, QED
    def __init__(self, dataset_name, df, raw_dir, split, randomize_order, MAX_LEN, scaling_type = 'std', 
        gflownet=False, gflownet_realgap=False, vocab=None, start_min=True, 
        scaler_std_properties=None, scaler_properties=None,  
        finetune_dataset=None, pretrain_dataset=None, 
        limited_properties=False, 
        train_split=0.98, val_split=0.01, test_split=0.01, force_prop_redo=False, 
        mask_props=[], extra_props_dont_always_mask=[], mask_seperately=False,
        remove_properties=None, portion_used=1.0,
        load_generated_mols=[], remove_unseen=True, reward_type="none"): 
        self.raw_dir = raw_dir
        self.dataset_name = dataset_name
        self.randomize_order = randomize_order
        self.MAX_LEN = MAX_LEN
        self.start_min = start_min
        self.vocab = vocab # needed because the column transformer will change the order of properties (categorical ones will be at the end)
        self.mask_prop = []
        self.extra_prop = []

        train_index, val_index, test_index = random_data_split(df.shape[0], self.raw_dir, self.dataset_name, train_split, val_split, test_split) # from seed 42
        if split == 'train':
            my_index = train_index
        elif split == 'valid':
            my_index = val_index
        elif split == 'test':
            my_index = test_index
        else:
            raise NotImplementedError()

        # if we plan to fine-tune, we must mask out the properties not included (there could be a better way)
        properties_finetune_names = np.array([], dtype=object)
        if finetune_dataset is not None:
            if finetune_dataset in ['bbbp', 'bace', 'hiv', 'zinc', 'qm9', 'moses']:
                pass
            elif finetune_dataset in ['xtb']:
                properties_finetune_names = np.array(['wavelength_energy','f_osc'], dtype=object)
        n_properties_finetune = len(properties_finetune_names)

        properties_pretrain_names = np.array([], dtype=object)
        if pretrain_dataset is not None:
            if pretrain_dataset in ['bbbp', 'bace', 'hiv', 'zinc', 'qm9', 'moses']:
                pass
            elif pretrain_dataset in ['xtb']:
                properties_pretrain_names = np.array(['wavelength_energy','f_osc'], dtype=object)
        n_properties_pretrain = len(properties_pretrain_names)

        self.categorical_prop = []
        which_to_keep = None
        if gflownet:
            assert self.dataset_name == 'qm9'
            self.smiles_list = df[my_index, 0]
            self.properties = np.expand_dims(df[my_index, 9].astype(float), axis=1)

            if gflownet_realgap:
                properties_path = os.path.join(self.raw_dir, f"properties_gflownet_nogap_{split}.npy")
                properties_ = np.load(properties_path).astype(float)
                self.properties = np.concatenate((self.properties, properties_), axis=1)
            else:
                properties_path = os.path.join(self.raw_dir, f"properties_gflownet_{split}.npy")
                real_gap = copy.deepcopy(self.properties[:, 0])
                self.properties = np.load(properties_path).astype(float)
                is_valid_path = os.path.join(self.raw_dir, f"is_valid_gflownet_{split}.npy")
                is_valid = np.load(is_valid_path)
                self.smiles_list = [smile for valid, smile in zip(is_valid, self.smiles_list) if valid] # mxnet humo-lumo gap is very finicky and doesnt work for some molecules
                
                # check and report the mse of real and mxmnet properties for the mxnet-valid molecules
                real_gap = real_gap[is_valid]
                fake_gap = self.properties[:, 0]
                mse = 0.5*((real_gap-fake_gap)**2).mean()
                print(f'MSE-mxmnet(split={split})={mse}')

        elif self.dataset_name in ['bbbp', 'bace', 'hiv']:
            self.categorical_prop = [0] # first variable is categorical
            self.smiles_list = df[my_index, 1] # col 1
            self.properties = np.concatenate((df[my_index, 0:1], df[my_index, 3:5]), axis=1).astype(float)
            self.properties_names = np.array(['hiv','SA','SE'], dtype=object)
        elif self.dataset_name in ['zinc_xtb']:
            self.smiles_list = df[my_index, 1]
            self.properties = None
        elif self.dataset_name in ['zinc', 'qm9']:
            self.smiles_list = df[my_index, 0] # col 0
            self.properties = None
        elif self.dataset_name in ['xtb']:
            self.properties = df[:, 2:4].astype(np.float32)
            self.properties_names = np.array(['wavelength_energy','f_osc'], dtype=object)
            self.smiles_list = df[:, 4] # col 0
            my_index = np.array(my_index)[~np.isinf(self.properties[my_index, 0])]
            self.properties = self.properties[my_index]
            self.smiles_list = self.smiles_list[my_index]

            if reward_type != 'none':
                if reward_type == "f_osc":
                    which_to_keep = self.properties[:,1]>4.5
                elif reward_type == "IR_f_osc":
                    which_to_keep = np.logical_and(self.properties[:,0]>=1000, self.properties[:,1]>0.1)
                else:
                    raise NotImplementedError()
                which_to_keep2_idx = np.random.choice(self.properties.shape[0], size=sum(which_to_keep), replace=False)
                which_to_keep2 = np.zeros(self.properties.shape[0], dtype=bool)
                which_to_keep2[which_to_keep2_idx] = True
                which_to_keep = which_to_keep | which_to_keep2

            self.properties[:,0] = 1239.8/self.properties[:,0] # change from lambda to energy

        # Mask nothing
        if self.properties is not None:
            self.mask_prop = self.mask_prop + [False for i in range(self.properties.shape[1])]
            self.extra_prop = self.extra_prop + [False for i in range(self.properties.shape[1])]

        # IF pretrain: [finetune(masked), current, extra]
        # ELIF finetune: [current, pretrain(masked), extra]
        # ElSE: [current, extra]

        if n_properties_finetune > 0: # append in front the finetune-data-specific properties
            properties_finetune = np.zeros((len(self.smiles_list), n_properties_finetune))
            if self.properties is None:
                self.properties = properties_finetune
                self.properties_names = properties_finetune_names
            else:
                self.properties = np.concatenate((properties_finetune, self.properties), axis=1)
                self.properties_names = np.concatenate((properties_finetune_names, self.properties_names))

            self.mask_prop = self.mask_prop + [True for i in range(n_properties_finetune)]
            self.extra_prop = self.extra_prop + [False for i in range(n_properties_finetune)]

        if n_properties_pretrain > 0: # append as second the pretrain-data-specific properties
            properties_pretrain = np.zeros((len(self.smiles_list), n_properties_pretrain))
            if self.properties is None:
                self.properties = properties_pretrain
                self.properties_names = properties_pretrain_names
            else:
                self.properties = np.concatenate((self.properties, properties_pretrain), axis=1)
                self.properties_names = np.concatenate((self.properties_names, properties_pretrain_names))
            self.mask_prop = self.mask_prop + [True for i in range(n_properties_pretrain)]
            self.extra_prop = self.extra_prop + [False for i in range(n_properties_pretrain)]

        if not limited_properties: # append as last the extra properties (included in both pretrain and finetune)
            print('3_prop')
            properties_path = os.path.join(self.raw_dir, f"{self.dataset_name}_3_properties_{split}.npy")
            properties_extra, properties_extra_names = load_or_create_3prop(self.smiles_list, properties_path, force_prop_redo=force_prop_redo)
            if self.properties is None:
                self.properties = properties_extra
                self.properties_names = properties_extra_names
            else:
                self.properties = np.concatenate((self.properties, properties_extra), axis=1)
                self.properties_names = np.concatenate((self.properties_names, properties_extra_names))
            self.mask_prop = self.mask_prop + [False for i in range(properties_extra.shape[1])]
            self.extra_prop = self.extra_prop + [True for i in range(properties_extra.shape[1])]

        n_added = 0
        for generate_mol_file in load_generated_mols: # not yet compatible with extra properties :(
            print(f'Appending generated molecule file: {generate_mol_file}')
            generated_mols = np.load(generate_mol_file, allow_pickle=True)
            # Removing duplicates
            #print(generated_mols[:,0])
            uniques, uniques_index = np.unique(generated_mols[:,0], return_index=True)
            if len(uniques) != len(generated_mols[:,0]):
                print('Found duplicates, deleting them and resaving')
                generated_mols = generated_mols[uniques_index]
                with open(generate_mol_file, 'wb') as f:
                    np.save(f, generated_mols)
            generated_smiles = generated_mols[:,0]
            generated_props = generated_mols[:,1:].astype(float)

            if remove_unseen: # Removing
                print('Verifying the generated SMILES for new tokens')
                assert vocab is not None # we must already have a vocabulary to check
                good_smiles = np.zeros(len(generated_smiles), dtype=bool)
                for i, gen_smile in enumerate(generated_smiles):
                    try:
                        gen_test = TargetData.from_smiles(gen_smile, vocab, randomize_order=randomize_order, MAX_LEN=MAX_LEN, start_min=not randomize_order).tokens
                        good_smiles[i] = True
                    except:
                        pass
                generated_smiles = generated_smiles[good_smiles]
                generated_props = generated_props[good_smiles]
                if len(good_smiles) != np.sum(good_smiles):
                    print(f'WARNING: We removed {len(good_smiles) - np.sum(good_smiles)}/{len(good_smiles)} molecules due to new tokens')
                else:
                    print('All generated SMILES were compatible.')

            # IF pretrain: [finetune(masked), current, extra]
            # ELIF finetune: [current, pretrain(masked), extra]
            # ElSE: [current, extra]
            if n_properties_finetune > 0: # append in front the finetune-data-specific properties
                generated_props = np.concatenate((properties_finetune, generated_props), axis=1)
            if n_properties_pretrain > 0: # append as second the pretrain-data-specific properties
                generated_props = np.concatenate((generated_props, properties_pretrain), axis=1)
            if not limited_properties: # append as last the extra properties (included in both pretrain and finetune)
                generate_mol_filename = os.path.split("/tmp/d/a.dat")[-1].split('.')[0]
                properties_path = os.path.join(self.raw_dir, f"{generate_mol_filename}_3_properties_{split}.npy")
                properties_extra, _ = load_or_create_3prop(generated_smiles, properties_path, force_prop_redo=force_prop_redo)
                generated_props = np.concatenate((generated_props, properties_extra), axis=1)

            assert generated_props.shape[1] == self.properties.shape[1] # must have the same number of properties
            n_added += generated_props.shape[0]
            self.smiles_list = np.concatenate((self.smiles_list, generated_smiles), axis=0)
            self.properties = np.concatenate((self.properties, generated_props), axis=0)
            if which_to_keep is not None:
                which_to_keep = np.concatenate((which_to_keep, np.ones(len(generated_smiles), dtype=bool)))
            print(f'Done appending generated molecule file')

        for prop in mask_props:
            idx_prop = np.where(prop == self.properties_names)[0][0]
            self.mask_prop[idx_prop] = True

        for prop in extra_props_dont_always_mask:
            idx_prop = np.where(prop == self.properties_names)[0][0]
            self.extra_prop[idx_prop] = False
        if not mask_seperately:
            for i in range(len(self.extra_prop)):
                self.extra_prop[i] = False

        if remove_properties is not None:
            for prop in remove_properties:
                idx_prop = np.where(prop == self.properties_names)[0][0]
                self.properties = np.delete(self.properties, idx_prop, axis=1)
                self.properties_names = np.delete(self.properties_names, idx_prop, axis=0)
                self.mask_prop.pop(idx_prop)
                self.extra_prop.pop(idx_prop)

        # Check and remove any inf/nans
        #print(self.properties[~np.isfinite(self.properties).all(axis=1)])
        good_ones = np.isfinite(self.properties).all(axis=1)
        self.properties = self.properties[good_ones]
        self.smiles_list = self.smiles_list[good_ones]
        self.n_properties = self.properties.shape[1]
        self.continuous_prop = [i for i in range(self.n_properties) if i not in self.categorical_prop]

        if scaler_std_properties is None:
            scaler_std = StandardScaler()
            self.scaler_std_properties = ColumnTransformer(scaler_std, self.continuous_prop, self.categorical_prop)
            self.scaler_std_properties.fit(self.properties)
        else:
            self.scaler_std_properties = scaler_std_properties

        self.scaling_type = scaling_type
        if scaler_properties is None:
            if scaling_type == 'std':   
                self.scaler_properties = self.scaler_std_properties
            elif scaling_type == 'minmax':
                scaler = MinMaxScaler(feature_range=(-1, 1))
                self.scaler_properties = ColumnTransformer(scaler, self.continuous_prop, self.categorical_prop)
                self.scaler_properties.fit(self.properties)
            elif scaling_type == 'none':
                self.scaler_properties = None
            else:
                raise NotImplementedError()
        else:
            self.scaler_properties = scaler_properties

        properties_std = self.scaler_std_properties.transform(self.properties)
        print(properties_std[0:2])
        self.mu_prior=np.mean(properties_std, axis=0)   
        self.cov_prior=np.cov(properties_std.T)

        if scaling_type == 'std':
            self.properties = properties_std
        elif self.scaler_properties is not None:
            self.properties = self.scaler_properties.transform(self.properties)

        if which_to_keep is not None:
            self.properties = self.properties[which_to_keep]
            self.smiles_list = self.smiles_list[which_to_keep]
        print(self.properties.shape)

        # Remove portion of the observations
        if portion_used < 1.0:
            indexes = np.random.choice(len(self.smiles_list), int(portion_used*len(self.smiles_list)), replace=False)
            self.smiles_list = self.smiles_list[indexes]
            if self.properties is not None:
                self.properties = self.properties[indexes]
        self.smiles_list = self.smiles_list.tolist()

    def update_vocab(self, vocab):
        self.vocab = vocab

    def update_smiles(self, smiles_list):
        self.smiles_list = smiles_list

    def update_properties(self, properties, scaler_std_properties=None, scaler_properties=None):
        self.properties = properties

        if scaler_std_properties is None:
            self.scaler_std_properties.fit(properties)
        else:
            self.scaler_std_properties = scaler_std_properties
        if scaler_properties is None and self.scaler_properties is not None:
            self.scaler_properties.fit(properties)
        else:
            self.scaler_properties = scaler_properties
        properties_std = self.scaler_std_properties.transform(properties)

        self.mu_prior=np.mean(properties_std, axis=0)
        self.cov_prior=np.cov(properties_std.T)

        if self.scaler_properties is not None:
            self.properties = self.scaler_properties.transform(self.properties)

    def get_mean_plus_std_property(self, idx, std):
        my_property_std = self.mu_prior[idx] + std*np.diag(self.cov_prior)[idx]
        properties = np.zeros((1, self.n_properties))
        properties[:, idx] = my_property_std
        if self.scaling_type != 'std':
            properties = self.scaler_std_properties.inverse_transform(properties) # std to raw
            if self.scaler_properties is not None:
                properties = self.scaler_properties.transform(properties) # raw to whatever
        return properties

    # Conditional probability for the properties
    # Modified from https://github.com/nyu-dl/conditional-molecular-design-ssvae/blob/master/SSVAE.py#L161
    def sampling_conditional_property(self, yid, ytarget):

        id2 = [yid]
        id1 = np.setdiff1d(np.arange(self.n_properties), id2)
    
        mu1 = self.mu_prior[id1]
        mu2 = self.mu_prior[id2]
        
        cov11 = self.cov_prior[id1][:,id1]
        cov12 = self.cov_prior[id1][:,id2]
        cov22 = self.cov_prior[id2][:,id2]
        cov21 = self.cov_prior[id2][:,id1]
        
        cond_mu = np.transpose(mu1.T+np.matmul(cov12, np.linalg.inv(cov22)) * (ytarget-mu2))[0]
        cond_cov = cov11 - np.matmul(np.matmul(cov12, np.linalg.inv(cov22)), cov21)
        
        marginal_sampled = np.random.multivariate_normal(cond_mu, cond_cov, 1)
        
        properties = np.zeros(self.n_properties)
        properties[id1] = marginal_sampled
        properties[id2] = ytarget
        
        if self.scaling_type != 'std':
            properties_notransf = self.scaler_std_properties.inverse_transform(properties.reshape(1, -1))
            if self.scaler_properties is not None:
                properties = self.scaler_properties.transform(properties_notransf).reshape(-1)
            else:
                properties = properties_notransf.reshape(-1)

        return properties

    def sampling_property(self, num_samples, scale=1):
        
        properties = np.random.multivariate_normal(self.mu_prior, self.cov_prior*scale, num_samples)
        
        if self.scaling_type != 'std':
            properties_notransf = self.scaler_std_properties.inverse_transform(properties.reshape(1, -1))
            if self.scaler_properties is not None:
                properties = self.scaler_properties.transform(properties_notransf).reshape(-1)
            else:
                properties = properties_notransf.reshape(-1)

        return properties


    def update_vocabs_scalers(self, dset_new):
        if self.scaling_type != 'none':
            properties = self.scaler_properties.inverse_transform(self.properties)
        else:
            properties = self.properties
        self.update_properties(properties, scaler_std_properties=dset_new.scaler_std_properties, scaler_properties=dset_new.scaler_properties)
        self.update_vocab(dset_new.vocab)

    def update_scalers(self, dset_new):
        if self.scaling_type != 'none':
            properties = self.scaler_properties.inverse_transform(self.properties)
        else:
            properties = self.properties
        self.update_properties(properties, scaler_std_properties=dset_new.scaler_std_properties, scaler_properties=dset_new.scaler_properties)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        #self.properties.tofile(os.path.join(self.raw_dir, 'train.bin'))
        smiles = self.smiles_list[idx]
        properties = self.properties[idx]
        return TargetData.from_smiles(smiles, self.vocab, randomize_order=self.randomize_order, MAX_LEN=self.MAX_LEN, start_min=self.start_min).featurize(), torch.from_numpy(properties).to(dtype=torch.float32)

def get_cond_datasets(dataset_name, raw_dir, randomize_order, MAX_LEN, scaling_type = 'std', 
    gflownet=False, gflownet_realgap=False, start_min=True, force_vocab_redo=False, force_prop_redo=False, sort=True, 
    finetune_dataset=None, pretrain_dataset=None, 
    limited_properties=False, 
    mask_props=[], extra_props_dont_always_mask=[], mask_seperately=False,
    remove_properties=None, portion_used=1.0,
    load_generated_mols=[], remove_unseen=True, reward_type="f_osc"):

    vocab_train_path = os.path.join(raw_dir, f"vocab_trainval.npy")
    vocab_test_path = os.path.join(raw_dir, f"vocab_trainvaltest.npy")
    if force_vocab_redo:
        vocab_train = None
        vocab_test = None
    else:
        try:
            vocab_train = load(vocab_train_path)
            vocab_test = load(vocab_test_path)
            print("Loaded the vocabulary")
        except:
            vocab_train = None
            vocab_test = None
            print("Could not load the vocabulary; building from scratch")

    if dataset_name in ['xtb']:
        df = pd.concat(map(pd.read_csv, sorted(glob.glob(os.path.join(raw_dir, "random_generation_stda_xtb_*.csv")))))
        df = df.values
    else:
        data_path = os.path.join(raw_dir, f"data.csv")
        df = pd.read_csv(data_path, sep=',')
        df = df.values

    train_dataset = PropCondDataset(dataset_name=dataset_name, raw_dir=raw_dir, split="train", randomize_order=randomize_order, 
        MAX_LEN=MAX_LEN, scaling_type=scaling_type, 
        gflownet=gflownet, gflownet_realgap=gflownet_realgap, 
        vocab=vocab_train, start_min=start_min, df=df, force_prop_redo=force_prop_redo,
        finetune_dataset=finetune_dataset, pretrain_dataset=pretrain_dataset, 
        limited_properties=limited_properties, 
        extra_props_dont_always_mask=extra_props_dont_always_mask, mask_seperately=mask_seperately, mask_props=mask_props,
        remove_properties=remove_properties, portion_used=portion_used,
        load_generated_mols=load_generated_mols, remove_unseen=remove_unseen, reward_type=reward_type)
    val_dataset = PropCondDataset(dataset_name=dataset_name, raw_dir=raw_dir, split="valid", randomize_order=False, 
        MAX_LEN=MAX_LEN, scaling_type=scaling_type, 
        gflownet=gflownet, gflownet_realgap=gflownet_realgap, 
        vocab=vocab_train, start_min=start_min, df=df, force_prop_redo=force_prop_redo,
        scaler_std_properties=train_dataset.scaler_std_properties, scaler_properties=train_dataset.scaler_properties,
        finetune_dataset=finetune_dataset, pretrain_dataset=pretrain_dataset, 
        limited_properties=limited_properties, 
        extra_props_dont_always_mask=extra_props_dont_always_mask, mask_seperately=mask_seperately, mask_props=mask_props,
        remove_properties=remove_properties, portion_used=portion_used,
        load_generated_mols=load_generated_mols, remove_unseen=remove_unseen, reward_type=reward_type) 
    test_dataset = PropCondDataset(dataset_name=dataset_name, raw_dir=raw_dir, split="test", randomize_order=False, 
        MAX_LEN=MAX_LEN, scaling_type=scaling_type, 
        gflownet=gflownet, gflownet_realgap=gflownet_realgap, 
        vocab=vocab_test, start_min=start_min, df=df, force_prop_redo=force_prop_redo,
        scaler_std_properties=train_dataset.scaler_std_properties, scaler_properties=train_dataset.scaler_properties,
        finetune_dataset=finetune_dataset, pretrain_dataset=pretrain_dataset, 
        limited_properties=limited_properties, 
        extra_props_dont_always_mask=extra_props_dont_always_mask, mask_seperately=mask_seperately, mask_props=mask_props,
        remove_properties=remove_properties, portion_used=portion_used,
        load_generated_mols=load_generated_mols, remove_unseen=remove_unseen, reward_type=reward_type)  

    if vocab_train is None:
        all_smiles_list = train_dataset.smiles_list + val_dataset.smiles_list + test_dataset.smiles_list

        print('smiles_list2atoms_list')
        # Create Vocabulary of atoms (we can use train, valid, test; even if we will never generate tokens from valid and test due to the training)
        TOKEN2ATOMFEAT, VALENCES = smiles_list2atoms_list(all_smiles_list, TOKEN2ATOMFEAT={}, VALENCES={}) 
        
        print('get_max_valence_from_dataset')
        # Calculate maximum Valency of atoms (we should get the valency from the training and validation data only)
        VALENCES = get_max_valence_from_dataset(train_dataset.smiles_list + val_dataset.smiles_list, VALENCES=VALENCES)
        # Make and update vocab
        vocab_train = SpanningTreeVocabulary(TOKEN2ATOMFEAT, VALENCES, sort=sort)
        dump(vocab_train, vocab_train_path)
        train_dataset.update_vocab(vocab_train)
        val_dataset.update_vocab(vocab_train)

        print('get_max_valence_from_dataset')
        # Calculate maximum Valency of atoms (with train, val, and test datasets)
        VALENCES_test = get_max_valence_from_dataset(all_smiles_list, VALENCES=copy.deepcopy(VALENCES))
        # Make and update vocab
        vocab_test = SpanningTreeVocabulary(TOKEN2ATOMFEAT, VALENCES_test, sort=sort)
        dump(vocab_test, vocab_test_path)
        test_dataset.update_vocab(vocab_test)

    print(f"Atom vocabulary size: {len(train_dataset.vocab.ATOM_TOKENS)}")
    print(train_dataset.vocab.ATOM_TOKENS)
    print("Valences from train + val")
    print(train_dataset.vocab.VALENCES)
    print("Valences from train + val + test")
    print(test_dataset.vocab.VALENCES)

    return train_dataset, val_dataset, test_dataset

def merge_datasets(dset, dset2, scaler_std_properties=None, scaler_properties=None):

    dset_new = copy.deepcopy(dset)

    # merge properties
    if dset.scaling_type != 'none':
        properties = dset.scaler_properties.inverse_transform(dset.properties)
        properties2 = dset2.scaler_properties.inverse_transform(dset2.properties)
    else:
        properties = dset.properties
        properties2 = dset2.properties
    assert properties.shape[1] == properties2.shape[1]
    all_properties = np.concatenate((properties, properties2), axis=0)
    dset_new.update_properties(all_properties, scaler_std_properties=scaler_std_properties, scaler_properties=scaler_properties)

    # merge smiles
    all_smiles_list = dset.smiles_list + dset2.smiles_list
    dset_new.update_smiles(all_smiles_list)

    # merge vocabs
    vocab_merged = merge_vocabs(dset.vocab, dset2.vocab)
    dset_new.update_vocab(vocab_merged)

    return dset_new

def get_spanning_tree_from_smiles(smiles_list, randomize_order=False, MAX_LEN=250, vocab=None):

    if vocab is None:
        # Create Vocabulary of atoms (we can use train, valid, test; even if we will never generate tokens from valid and test due to the training)
        TOKEN2ATOMFEAT, VALENCES = smiles_list2atoms_list_multiprocess(smiles_list, TOKEN2ATOMFEAT={}, VALENCES={}) 
        # Calculate maximum Valency of atoms (we should get the valency from the training and validation data only)
        VALENCES = get_max_valence_from_dataset_multiprocess(smiles_list, VALENCES=VALENCES)
        # Make and update vocab
        vocab = SpanningTreeVocabulary(TOKEN2ATOMFEAT, VALENCES)

    out = []
    for smile in smiles_list:
        out += ["".join(TargetData.from_smiles(smile, vocab, randomize_order=randomize_order, MAX_LEN=MAX_LEN, start_min=not randomize_order).tokens)]

    if len(out) == 1:
        return out[0]
    else:
        return out