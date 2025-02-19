# Adapted from https://github.com/wengong-jin/hgraph2graph/

import torch
import rdkit
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdmolops, Descriptors, rdMolDescriptors 
from rdkit.Chem import Descriptors
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed
import networkx as nx
import props.sascorer as sascorer
from moses.utils import mapper
import torch_geometric.data as gd
from rdkit.rdBase import BlockLogs
from model import mxmnet
import numpy as np
from props.xtb.stda_xtb import STDA_XTB, default_stda_config
import pandas as pd
import os
import multiprocessing

def similarity(a, b):
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def penalized_logp(s):
    mol = Chem.MolFromSmiles(s)

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle


def MolLogP_mols_(mol):
    RDLogger.DisableLog('rdApp.*')
    return MolLogP(mol)
def MolLogP_mols(mols, num_workers=6):
    return [MolLogP_mols_(mol) for mol in mols]

def qed_mols_(mol):
    RDLogger.DisableLog('rdApp.*')
    return qed(mol)
def qed_mols(mols, num_workers=6):
    return [qed_mols_(mol) for mol in mols]

def ExactMolWt_mols_(mol):
    RDLogger.DisableLog('rdApp.*')
    return Descriptors.ExactMolWt(mol)
def ExactMolWt_mols(mols, num_workers=6):
    return [ExactMolWt_mols_(mol) for mol in mols]


def MolLogP_smiles(smiles, num_workers=6):
    return [MolLogP_mols_(Chem.MolFromSmiles(smile)) for smile in smiles]
def qed_smiles(smiles, num_workers=6):
    return [qed_mols_(Chem.MolFromSmiles(smile)) for smile in smiles]
def ExactMolWt_smiles(smiles, num_workers=6):
    return [ExactMolWt_mols_(Chem.MolFromSmiles(smile)) for smile in smiles]

def MAE_properties(mols, properties, properties_estimated=None, name="none", max_abs_value=None): # molwt, LogP, QED

    gen_properties = None

    if name == "ExactMolWt":
        gen_molwt = ExactMolWt_mols(mols)
        gen_molwt = torch.tensor(gen_molwt).to(dtype=properties.dtype, device=properties.device).unsqueeze(1)
        gen_properties = gen_molwt
    elif name == "MolLogP":
        gen_logp = MolLogP_mols(mols)
        gen_logp = torch.tensor(gen_logp).to(dtype=properties.dtype, device=properties.device).unsqueeze(1)
        if gen_properties is None:
            gen_properties = gen_logp
        else:
            gen_properties = torch.cat([gen_properties, gen_logp], dim=1)
    elif name == "QED":
        gen_qed = qed_mols(mols)
        gen_qed = torch.tensor(gen_qed).to(dtype=properties.dtype, device=properties.device).unsqueeze(1)
        if gen_properties is None:
            gen_properties = gen_qed
        else:
            gen_properties = torch.cat([gen_properties, gen_qed], dim=1)
    else:
        raise NotImplementedError()

    losses = (gen_properties-properties).abs().mean(dim=1)
    if properties_estimated is not None:
        losses_estimated = (gen_properties - properties_estimated).abs().mean(dim=1)
    else:
        losses_estimated = None
    n = gen_properties.shape[0]
    losses_1, index_best = torch.topk(losses, min(1, n), dim=0, largest=False, sorted=True)
    losses_10, index_best10 = torch.topk(losses, min(10, n), dim=0, largest=False, sorted=True)
    losses_100, index_best100 = torch.topk(losses, min(100, n), dim=0, largest=False, sorted=True)
    if n < 100:
        print(f'Warning: Less than 100 valid molecules ({n} valid molecules), so results will be poor')
    Min_MAE = losses_1[0]
    print(Chem.MolToSmiles(mols[index_best])) # best molecule
    Min10_MAE = torch.mean(losses_10)
    Min100_MAE = torch.mean(losses_100)
    if max_abs_value is not None: # get percentage error
        Min_MAE = Min_MAE / max_abs_value
        Min10_MAE = Min10_MAE / max_abs_value
        Min100_MAE = Min100_MAE / max_abs_value
    
    if properties_estimated is not None:
        return Min_MAE, Min10_MAE, Min100_MAE, losses_estimated[index_best].mean(0), losses_estimated[index_best10].mean(0), losses_estimated[index_best100].mean(0)
    else:
        return Min_MAE, Min10_MAE, Min100_MAE, 0.0, 0.0, 0.0

def remove_duplicates(generated_smiles, train_smiles):
    train_smiles_canon = [Chem.CanonSmiles(smile) for smile in train_smiles]
    generated_smiles_canon = [Chem.CanonSmiles(smile) for smile in generated_smiles]
    non_dup_smiles = []
    for i in range(len(generated_smiles)):
        smile_canon = generated_smiles_canon[i]
        all_other_smiles = train_smiles_canon + generated_smiles_canon[0:i] + generated_smiles_canon[(i+1):]
        if smile_canon not in all_other_smiles:
            non_dup_smiles += [True]
        else:
            non_dup_smiles += [False]
    return non_dup_smiles

def return_top_k(properties, properties_estimated, k=100, mask_cond=None,
    props_must_min=None, props_must_max=None):
    if mask_cond is not None:
        properties_ = properties[:, ~mask_cond]
        properties_estimated_ = properties_estimated[:, ~mask_cond]
    else:
        properties_ = properties
        properties_estimated_ = properties_estimated
    losses = 0
    c = properties_.shape[1]
    for i in range(c):
        if props_must_min[i]:
            losses += properties_estimated_[:, i] / c
        elif props_must_max[i]:
            losses += -properties_estimated_[:, i] / c
        else:
            losses += ((properties_[:, i]-properties_estimated_[:, i])/properties_[:, i]).abs() / c
    losses, index_best = torch.topk(losses, min(k, properties_.shape[0]), dim=0, largest=False, sorted=True)
    return index_best

def MAE_properties_estimated(mols, train_smiles, properties, properties_estimated, mask_cond=None, 
    zero_rank=True, properties_all=None, properties_names=None,
    props_must_min=None, props_must_max=None, top_k=100, store_output="", oracle_samples=0):

    if mask_cond is not None:
        properties_ = properties[:, ~mask_cond]
        properties_estimated_ = properties_estimated[:, ~mask_cond]
    else:
        properties_ = properties
        properties_estimated_ = properties_estimated
    losses = 0
    c = properties_.shape[1]
    for i in range(c):
        if props_must_min[i]:
            losses += properties_estimated_[:, i] / c
        elif props_must_max[i]:
            losses += -properties_estimated_[:, i] / c
        else:
            losses += ((properties_[:, i]-properties_estimated_[:, i])/properties_[:, i]).abs() / c
    n = properties.shape[0]
    losses_1, index_best = torch.topk(losses, 1, dim=0, largest=False, sorted=True)
    losses_10, index_best10 = torch.topk(losses, min(10, n), dim=0, largest=False, sorted=True)
    losses_100, index_best100 = torch.topk(losses, min(100, n), dim=0, largest=False, sorted=True)
    losses_topk, index_besttopk = torch.topk(losses, min(top_k, n), dim=0, largest=False, sorted=True)
    if n < 100:
        print(f'Warning: Less than 100 valid molecules ({n} valid molecules), so results will be poor')
    Min_MAE = losses_1[0]
    Min10_MAE = torch.mean(losses_10)
    Min100_MAE = torch.mean(losses_100)

    similarities, smiles_closests, smiles_closests2, smiles_closests3 = average_similarity([mols[ind] for ind in index_besttopk], train_smiles, device=properties.device)

    Min_average_similarity = similarities[0]
    Min10_average_similarity = torch.mean(similarities[0:10])
    Min100_average_similarity = torch.mean(similarities[0:100])

    Min10_diversity = top_k_diversity([mols[ind] for ind in index_best10], K=10, device=properties.device)

    if zero_rank:
        if store_output != "":
            Min10_diversity_ = np.repeat(Min10_diversity.cpu().numpy(), repeats=min(top_k, n))
            N_samples = np.repeat(oracle_samples, repeats=min(top_k, n))
            losses_topk_np = losses_topk.cpu().numpy()
            similarities_np = similarities.cpu().numpy()
            top_mols = [Chem.MolToSmiles(mols[ind]) for ind in index_besttopk]
            x_comb = np.stack((N_samples, losses_topk_np,top_mols,Min10_diversity_,similarities_np,smiles_closests,smiles_closests2,smiles_closests3), axis=1)
            columns=['N','loss','smiles','diversity-10','closest_smiles_sim', 'closest_smiles', '2nd_closest_smiles', '3rd_closest_smiles']
            if properties_names is not None and properties_all is not None:
                properties_all_ = properties_all[index_besttopk].cpu().numpy()
                x_comb = np.concatenate((x_comb,properties_all_), axis=1)
                columns += properties_names.tolist()
            data = pd.DataFrame(data=x_comb, columns=columns)
            data.to_csv(store_output)
        print('Random 25 molecules')
        print([Chem.MolToSmiles(mols[ind]) for ind in np.random.randint(n, size=25)])
        print('Top-k molecules')
        print([Chem.MolToSmiles(mols[ind]) for ind in index_besttopk]) # best molecules
        print('similarities of the closests molecules in the training data')
        print(similarities)
        print('Closest molecules in the training data')
        print(smiles_closests)
        print('2nd-Closest molecules in the training data')
        print(smiles_closests2)
        print('3rd-Closest molecules in the training data')
        print(smiles_closests3)
        print('Asked properties')
        print(properties[0])
        print('Top-k properties')
        if properties_names is not None:
            print(properties_names)
        if properties_all is not None:
            print(properties_all[index_besttopk])
        else:
            print(properties_estimated[index_besttopk])

    return Min_MAE, Min10_MAE, Min100_MAE, Min_average_similarity, Min10_average_similarity, Min100_average_similarity, Min10_diversity

def get_xtb_scores(smiles, sqrt=False, name="name"):
    if not os.path.exists(f"CHANGE_TO_YOUR_DIR/stda_scratch/{name}"): 
        os.makedirs(f"CHANGE_TO_YOUR_DIR/stda_scratch/{name}")
    if not os.path.exists(f"CHANGE_TO_YOUR_DIR/mol_coords_logs/{name}"): 
        os.makedirs(f"CHANGE_TO_YOUR_DIR/mol_coords_logs/{name}")
    if not os.path.exists(f"CHANGE_TO_YOUR_DIR/final_mol_coords/{name}"): 
        os.makedirs(f"CHANGE_TO_YOUR_DIR/final_mol_coords/{name}")
    cfg = {
        'log_dir': f"CHANGE_TO_YOUR_DIR/stda_scratch/{name}",
        'xtb_path': "CHANGE_TO_YOUR_DIR/xtb4stda", 
        'stda_command': "stda_v1.6.3",
        "moltocoord_config": {
            'log_dir': f"CHANGE_TO_YOUR_DIR/mol_coords_logs/{name}",
            "final_coords_log_dir": f"CHANGE_TO_YOUR_DIR/final_mol_coords/{name}", # save all final coordinates to here
            'ff': 'GFN-FF',  # or MMFF, UFF, or RDKIT (generates by ETKDG)
            'semipirical_opt': 'xtb',  # or None
            'conformer_config': {
                "num_conf": 10,
                "maxattempts": 100,
                "randomcoords": True,
                "prunermsthres": 1.5,
            },
        },
        'stda_cutoff': 6,
        "remove_scratch": True,
    }
    stdaxtb = STDA_XTB(**cfg)
    props = stdaxtb.get_scores(smiles)
    if sqrt:
        props[:,1] = np.sqrt(props[:,1])
    return props

# From https://arxiv.org/pdf/2210.12765

def compute_other_flat_rewards(mol):
    logp = np.exp(-((Descriptors.MolLogP(mol) - 2.5)**2) / 2)
    sa = (10 - sascorer.calculateScore(mol)) / 9  # Turn into a [0-1] reward
    molwt = np.exp(-((Descriptors.MolWt(mol) - 105)**2) / 150)
    return logp, sa, molwt

@torch.no_grad()
def compute_flat_rewards(mols, device, gap_model=None):
    assert len(mols) <= 128

    if gap_model is None:
        gap_model = mxmnet.MXMNet(mxmnet.Config(128, 6, 5.0))
        state_dict = torch.load('../resource/data/mxmnet_gap_model.pt')
        gap_model.load_state_dict(state_dict)
        gap_model.to(device)

    other_flats = torch.as_tensor(
        [compute_other_flat_rewards(mol) for mol in mols]).float().to(device)

    graphs = [mxmnet.mol2graph(i) for i in mols]
    is_valid = [graph is not None for graph in graphs]
    graphs = [graph for graph in graphs if graph is not None]
    batch = gd.Batch.from_data_list(graphs)
    batch.to(device)
    preds = gap_model(batch).reshape((-1,)).data.cpu() / mxmnet.HAR2EV
    preds[preds.isnan()] = 0 # they used 1 in gflownet, but this makes no sense and it could inflate the predicted property
    preds_mxnet = torch.as_tensor(preds).float().to(device).clip(1e-4, 2).reshape((-1, 1))
    flat_rewards = torch.cat([preds_mxnet, other_flats[is_valid]], 1)
    return flat_rewards

def compute_other_flat_properties(mol):
    logp = Descriptors.MolLogP(mol)
    sa = sascorer.calculateScore(mol)
    molwt = Descriptors.MolWt(mol)
    return logp, sa, molwt

@torch.no_grad()
def compute_flat_properties(smiles, device, num_workers=24):
    gap_model = mxmnet.MXMNet(mxmnet.Config(128, 6, 5.0))
    state_dict = torch.load('../resource/data/mxmnet_gap_model.pt')
    gap_model.load_state_dict(state_dict)
    gap_model.to(device)

    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    # Get properties
    other_flats = torch.as_tensor(
        [compute_other_flat_properties(mol) for mol in mols]).float().to(device)
    preds_mxnet = None
    offset = 0
    num_samples = len(mols)
    while offset < num_samples:
        cur_num_samples = min(num_samples - offset, 128)
        #graphs = mapper(num_workers)(mxmnet.mol2graph, mols[offset:(offset+cur_num_samples)])
        graphs = [mxmnet.mol2graph(mol) for mol in mols[offset:(offset+cur_num_samples)]]
        is_valid_ = [graph is not None for graph in graphs]
        graphs = [graph for graph in graphs if graph is not None]

        batch = gd.Batch.from_data_list(graphs)
        batch.to(device)
        preds = gap_model(batch).reshape((-1,)).data.cpu() / mxmnet.HAR2EV
        offset += cur_num_samples
        print(offset)
        preds[preds.isnan()] = 1 # weird, but thats what the multi-objective gflownet paper does, but there is never any nan in the dataset (but there can be in the generated ones)
        preds = torch.as_tensor(preds).float().to(device).clip(1e-4, 2).reshape((-1, 1))
        if preds_mxnet is None:
            preds_mxnet = preds
            is_valid = is_valid_
        else:
            preds_mxnet = torch.cat((preds_mxnet, preds), dim=0)
            is_valid += is_valid_
    flat_properties = torch.cat([preds_mxnet, other_flats[is_valid]], 1)
    return is_valid, flat_properties.cpu().detach().numpy()

@torch.no_grad()
def compute_flat_properties_nogap(smiles, device, num_workers=24):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    flat_properties = torch.as_tensor(
        [compute_other_flat_properties(mol) for mol in mols]).float().to(device)
    return flat_properties.cpu().detach().numpy()

# Taken from https://github.com/recursionpharma/gflownet/blob/f106cdeb6892214cbb528a3e06f4c721f4003175/src/gflownet/utils/metrics.py#L584
def top_k_diversity(mols, K=10, device=None):
    fps = [Chem.RDKFingerprint(mol) for mol in mols]
    x = []
    for i, y in enumerate(fps):
        if y is None:
            raise NotImplementedError()
            #continue
        x.append(y)
        if len(x) >= K:
            break
    s = np.array([DataStructs.BulkTanimotoSimilarity(i, x) for i in x])
    return torch.tensor((np.sum(s) - len(x)) / (len(x) * len(x) - len(x)), device=device)  # substract the diagonal

def get_fingerprint(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:  # Handle cases where SMILES can't be parsed
        return None
    fp = Chem.RDKFingerprint(mol)
    return fp

def average_similarity2(gen_smiles, smiles, device=None, num_workers=24):
    with multiprocessing.Pool(processes=num_workers) as pool:
        fps = pool.map(get_fingerprint, gen_smiles)
    fps = [fp for fp in fps]
    print(gen_smiles)
    print(fps)
    with multiprocessing.Pool(processes=num_workers) as pool:
        fps_train = pool.map(get_fingerprint, smiles)
    fps_train = [fp for fp in fps_train if fp is not None]
    #fps = mapper(24)(get_fingerprint, gen_smiles)
    #fps_train = mapper(24)(get_fingerprint, smiles)
    #fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(smile)) for smile in gen_smiles]
    #fps_train = [Chem.RDKFingerprint(Chem.MolFromSmiles(smile)) for smile in smiles]
    sims_out = []
    smiles_closests = []
    smiles_closests2 = []
    smiles_closests3 = []
    print("-- Getting similarities --")
    for i, (fp, gen_smile) in enumerate(zip(fps, gen_smiles)):
        print(gen_smile)
        if fp is None:
            print("No fingerprint")
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps_train)
        sims_3, indexes_3 = torch.topk(torch.tensor(sims), min(3, len(sims)), dim=0, largest=True, sorted=True)
        print(sims_3[0])
        sims_out.append(sims_3[0])
        smiles_closests.append(smiles[indexes_3[0]])
        if len(sims) >=2:
            smiles_closests2.append(smiles[indexes_3[1]])
        else:
            smiles_closests2.append(None)
        if len(sims) >=3:
            smiles_closests3.append(smiles[indexes_3[2]])
        else:
            smiles_closests3.append(None)
    return torch.tensor(sims_out, device=device), smiles_closests, smiles_closests2, smiles_closests3

def average_similarity(gen_mols, smiles, device=None):
    fps = [Chem.RDKFingerprint(mol) for mol in gen_mols]
    fps_train = [Chem.RDKFingerprint(Chem.MolFromSmiles(smile)) for smile in smiles]
    sims_out = []
    smiles_closests = []
    smiles_closests2 = []
    smiles_closests3 = []
    for i, y in enumerate(fps):
        if y is None:
            raise NotImplementedError()
            #continue
        sims = DataStructs.BulkTanimotoSimilarity(y, fps_train)
        sims_3, indexes_3 = torch.topk(torch.tensor(sims), min(3, len(sims)), dim=0, largest=True, sorted=True)
        sims_out.append(sims_3[0])
        smiles_closests.append(smiles[indexes_3[0]])
        if len(sims) >=2:
            smiles_closests2.append(smiles[indexes_3[1]])
        else:
            smiles_closests2.append(None)
        if len(sims) >=3:
            smiles_closests3.append(smiles[indexes_3[2]])
        else:
            smiles_closests3.append(None)
    return torch.tensor(sims_out, device=device), smiles_closests, smiles_closests2, smiles_closests3

def average_similarity_(gen_smiles, fps_train, device=None):
    fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(smile)) for smile in gen_smiles]
    sims_out = []
    for i, y in enumerate(fps):
        if y is None:
            raise NotImplementedError()
            #continue
        sims = DataStructs.BulkTanimotoSimilarity(y, fps_train)
        sims_top, _ = torch.topk(torch.tensor(sims), 1, dim=0, largest=True, sorted=True)
        sims_out.append(sims_top[0])
    return torch.tensor(sims_out, device=device)

def best_rewards_gflownet(smiles, mols, device): # molwt, LogP, QED

    gap_model = mxmnet.MXMNet(mxmnet.Config(128, 6, 5.0))
    state_dict = torch.load('../resource/data/mxmnet_gap_model.pt')
    gap_model.load_state_dict(state_dict)
    gap_model.to(device)

    gen_rewards = None
    offset = 0
    num_samples = len(mols)
    while offset < num_samples:
        cur_num_samples = min(num_samples - offset, 128)
        gen_rewards_ = compute_flat_rewards(mols[offset:(offset+cur_num_samples)], device, gap_model=gap_model)
        offset += cur_num_samples
        print(offset)
        if gen_rewards is None:
            gen_rewards = gen_rewards_
        else:
            gen_rewards = torch.cat((gen_rewards_, gen_rewards), dim=0)
    gen_rewards_mean = gen_rewards.mean(dim=1)
    n = gen_rewards.shape[0]
    rewards_10, indexes_10 = torch.topk(gen_rewards_mean, min(10, n), dim=0, largest=True, sorted=True)
    top_rewards = gen_rewards[indexes_10, :].mean(0)
    top_rewards_mean = top_rewards.mean()
    diversity = top_k_diversity(mols[indexes_10], K=10)

    top_rewards_weighted = 0
    diversity_weighted = 0
    for i in range(10):
        torch.manual_seed(i) # deterministic
        w = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.5] * 4)).sample().to(device)
        gen_rewards_weighted = (w*gen_rewards).sum(dim=1)
        n = gen_rewards.shape[0]
        rewards_10, indexes_10_ = torch.topk(gen_rewards_weighted, min(10, n), dim=0, largest=True, sorted=True)
        top_rewards_weighted += rewards_10.mean() / 10
        diversity_weighted += top_k_diversity(mols, gen_rewards_weighted.cpu().numpy(), K=10) / 10

    return top_rewards_weighted, diversity_weighted, top_rewards_mean, diversity, top_rewards[0], top_rewards[1], top_rewards[2], top_rewards[3], [smiles[i] for i in indexes_10]


if __name__ == "__main__":
    print(
        round(
            penalized_logp("ClC1=CC=C2C(C=C(C(C)=O)C(C(NC3=CC(NC(NC4=CC(C5=C(C)C=CC=C5)=CC=C4)=O)=CC=C3)=O)=C2)=C1"), 2
        ),
        5.30,
    )
