import multiprocessing, os, string, shutil, subprocess, traceback
import ray
import random
from rdkit import Chem
import pandas as pd
from rdkit.Chem import AllChem
from chem import get_rdkit_ff_coordinates, get_conformers, run_gfn_ff, run_gfn_xtb

#@ray.remote
class MolToCoord:
    def __init__(self, config):
        self.log_dir = config.get("log_dir")
        os.makedirs(self.log_dir, exist_ok=True)

        self.ff = config.get("ff", "MMFF")

        self.conformer_config = config.get("conformer_config")
        self.semiempirical_opt = config.get("semipirical_opt")
        self.semiempirical_config = config.get("semiempirical_config", {})

        self.final_coords_log_dir = config.get("final_coords_log_dir", None)
        if self.final_coords_log_dir is not None:
            os.makedirs(self.final_coords_log_dir, exist_ok=True)

    def _conformer_wrapper(self, mol, save_dir, mol_name):
        print('generating conformers for {}'.format(mol_name))
        get_conformers(mol=mol, filename=mol_name, filepath=save_dir, **self.conformer_config)
        return None

    def _rdkit_wrapper(self, mol, save_dir, mol_name):
        print('converting to ETKDG coordinates via RDKIT for {}'.format(mol_name))
        Chem.SanitizeMol(mol)
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h)
        coord_file = os.path.join(save_dir, mol_name + '_' + str(self.ff) + '_opt.xyz')
        Chem.MolToXYZFile(mol_h, coord_file, confId=0)  # save xyz
        return coord_file

    def _rdkit_ff_wrapper(self, 
        mol, save_dir, mol_name):
        print('converting to {} coordinates for {}'.format(str(self.ff), mol_name))
        _, coord_file = get_rdkit_ff_coordinates(mol, self.ff, conformer_config=self.conformer_config, filepath=save_dir, filename=mol_name)
        return coord_file

    def _gfn_ff_wrapper(self, mol, save_dir, mol_name):
        print('converting to gfn_ff coordinates for {}'.format(mol_name))
        coord_file = os.path.join(save_dir, mol_name + '_' + str(self.ff) + '_opt.xyz')
        run_gfn_ff(filepath=save_dir, filename=mol_name + '_' + str(self.ff), mols=mol, conformer_config=self.conformer_config)
        if os.path.isfile(os.path.join(save_dir, mol_name + '_' + str(self.ff) + '_lowest_energy_gfnffopt.xyz')): 
            subprocess.run("mv {} {}".format(os.path.join(save_dir, mol_name + '_' + str(self.ff) + '_lowest_energy_gfnffopt.xyz'), coord_file), shell=True, executable="/bin/bash")
        else:
            print('GFN-FF coord failed')
        return coord_file

    def _gfn_xtb_wrapper(self, save_dir, input_coord):
        print('converting to gfn_xtb coordinates for {}'.format(input_coord))
        run_gfn_xtb(filepath=save_dir, filename=input_coord, log_file_dir='log_gfn_xtb',
                    optimized_xyz_dir=save_dir, **self.semiempirical_config)

    def _ff_geometry(self, mol, save_dir, mol_name):
        if self.ff == 'UFF' or self.ff == 'MMFF':
            ff_coord_file = self._rdkit_ff_wrapper(mol, save_dir, mol_name)
        elif self.ff == 'GFN-FF':
            ff_coord_file = self._gfn_ff_wrapper(mol, save_dir, mol_name)
        elif self.ff == 'RDKIT' or self.ff == 'ETKDG':
            ff_coord_file = self._rdkit_wrapper(mol, save_dir, mol_name)
        elif self.ff == 'Conformers':
            ff_coord_file = self._conformer_wrapper(mol, save_dir, mol_name)
        else:
            raise NotImplementedError
        print('ff geometry conversion for {} is done'.format(mol_name))
        return ff_coord_file

    def _semiempirical_geometry(self, save_dir, input_coord, final_coord):  # , optimized_xyz_dir='log_stda_xtb'
        if self.semiempirical_opt == 'xtb':
            self._gfn_xtb_wrapper(save_dir, input_coord)
            shutil.copy2(os.path.join(save_dir, input_coord.split('.')[0] + '_xtb_opt.xyz'),
                         os.path.join(save_dir, final_coord))
        elif self.semiempirical is None:
            shutil.copy2(os.path.join(save_dir, input_coord.split('.')[0] + '_xtb_opt.xyz'),
                         os.path.join(save_dir, final_coord))
        else: raise NotImplementedError
        print('semiempirical geometry conversion for {} is done'.format(input_coord))

    def optimize_molecule(self, mol, mol_name=None, *args, **kwargs):
        mol_name = mol_name or ''.join(random.choices(string.ascii_uppercase + string.digits, k=30))
        os.chdir(self.log_dir)
        save_dir = os.path.join(self.log_dir, mol_name)
        # print('save_dir is', save_dir)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        final_coord = mol_name + '_opt.xyz'

        if type(mol) is str:
            mol = Chem.MolFromSmiles(mol)
        elif os.path.isfile(mol):
            # the mol given is actually a coordinate file, in this case, just copy the file to the save_dir
            # todo optimization based on given mol file
            shutil.copy2(mol, os.path.join(save_dir, final_coord))
            return save_dir, final_coord
        elif type(mol) is not (Chem.rdchem.Mol or Chem.rdchem.EditableMol):
            raise NotImplementedError
        # TODO: take in raw coordinates

        # get initial geometry by gfn-ff, mmff, uff, or rdkit
        try:
            os.chdir(save_dir)

            ff_coord_file = self._ff_geometry(mol, save_dir, mol_name)

            # get final geometry by xtb, or just ff geometry
            if ff_coord_file is None:
                print('only conformers are saved, semi-empirical calculations are not carried out')
                return save_dir, None

            self._semiempirical_geometry(save_dir, ff_coord_file, final_coord)

            # if we want a copy of all final coordinates in the same folder

            if self.final_coords_log_dir is not None:
                shutil.copy2(os.path.join(save_dir, final_coord),
                             os.path.join(self.final_coords_log_dir, final_coord))
            os.chdir(self.log_dir)
            return save_dir, final_coord
        except Exception:
            print('Traceback optimize_molecule')
            # other parts already deal with: if one ff_opt fails; if semi-empirical opt fails; retry with conf embedding
            # occasionally, however, all embedding fails, or every ff geometry fails, and we need to catch this
            traceback.print_exc()
            return None, None

    #@ray.method(num_returns=2)
    def __call__(self, mol, mol_name, *args, **kwargs):
        return self.optimize_molecule(mol, mol_name)

# example config
default_moltocoord_config={
    "memory": int(1e+9), # 10GB
    "object_store_memory": int(1e+9), # 10GB
    "mols": "mols.csv",
    'log_dir': 'CHANGE_TO_YOUR_DIR/mol_coords_logs',
    "final_coords_log_dir": 'CHANGE_TO_YOUR_DIR/final_mol_coords', # save all final coordinates to here

    'ff': 'MMFF',  # or MMFF, UFF, or RDKIT (generates by ETKDG)
    'semipirical_opt': 'xtb',  # or None
    'conformer_config': {
        "num_conf": 8,
        "maxattempts": 100,
        "randomcoords": True,
        "prunermsthres": 1.5,
    },
}
if __name__ == "__main__":
    config = default_moltocoord_config
    # os.environ['RAY_worker_register_timeout_seconds'] = '30'
    print("initializing ray")
    ray.init(object_store_memory=config["object_store_memory"], _memory=config["memory"], logging_level='debug')
    print('initialized ray')
    mols = pd.read_csv(config["mols"])
    # for an actor pool
    actors = [MolToCoord.remote(config) for _ in range(multiprocessing.cpu_count())]
    pool = ray.util.ActorPool(actors)
    for i, mol in enumerate(mols['smiles']):
        print(mol)
        pool.submit(lambda actor, mol_config: actor.optimize_molecule.remote(**mol_config), {"mol":mol, "mol_name":str(i)})
    while pool.has_next():
        _, _ = pool.get_next_unordered()