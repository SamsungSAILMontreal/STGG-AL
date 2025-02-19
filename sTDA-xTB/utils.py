from rdkit import Chem, RDLogger
import torch
import glob
import re
import os
import numpy as np
import torch.nn.functional as F
bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        smiles = Chem.MolToSmiles(mol)
    except:
        return None   


    if len(smiles) == 0:
        return None

    return smiles

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def get_files_in_dir(dir, specs=None):
    if specs is None:
        return natural_sort(glob.glob(os.path.join(dir,"*")))
    else:
        return natural_sort(glob.glob(os.path.join(dir,specs)))
