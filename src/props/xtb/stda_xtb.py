import multiprocessing, os, string, shutil, subprocess, traceback
from dotmap import DotMap
import pandas as pd
import subprocess
import numpy as np
from utils.mol_to_coord import MolToCoord
import multiprocessing

'''
To run, install xtb via conda, or via source (+ export PATH="~/xtb/bin/:$PATH"). Download stda-xtb via source, and provide path below (can be inserted in .env file instead)

To distribute the number of threads reasonable in the OpenMP section for XTB it is recommended to use 
export OMP_NUM_THREADS=<ncores>,1
export OMP_STACKSIZE=4G

Currently using ray for mol_to_coord, and I recommend a ray wrapper for virtual screening a lot of STDA_XTB calculations.
'''

default_stda_config = {
    'log_dir': "CHANGE_TO_YOUR_DIR/stda_scratch",
    'xtb_path': "CHANGE_TO_YOUR_DIR/xtb4stda", 
    'stda_command': "stda_v1.6.3",
    "moltocoord_config": {
        'log_dir': "CHANGE_TO_YOUR_DIR/mol_coords_logs",
        "final_coords_log_dir": "CHANGE_TO_YOUR_DIR/final_mol_coords", # save all final coordinates to here
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
default_moltocoord_config={
    "memory": int(1e+9), # 10GB
    "object_store_memory": int(1e+9), # 10GB
}

class STDA_XTB:
    # currently only supports organic molecules with no charges, i.e. no periodic structures or charged molecules
    def __init__(self, xtb_path, log_dir, stda_command='stda_v1.6.3', moltocoord_config={},
                 scoring_function='wavelength+fosc', stda_cutoff=None, remove_scratch=True):

        self.xtb_path = xtb_path
        self.stda_command = stda_command

        self.stda_cutoff = stda_cutoff or 6 # stop calculating transitions above 6 eV, change to 10 for UV absorbers
        self.scoring_function = scoring_function

        self.log_dir = log_dir
        self.moltocoord_config = moltocoord_config
        self.moltocoord_config["log_dir"] = self.log_dir
        #self.moltocoord = MolToCoord.remote(self.moltocoord_config)

        self.remove_scratch = remove_scratch

        self.current_dir = os.getcwd()

    def __call__(self, molecule, mol_name=None, *args, **kwargs):
        return self.get_score(molecule, mol_name, **kwargs)

    def get_score(self, molecule, mol_name=None, save_dir=None, final_coord=None, **kwargs):
        os.chdir(self.log_dir)

        # get optimized coordinates via ff/semiempirical methods
        if save_dir is None or final_coord is None:
            save_dir, final_coord = MolToCoord(self.moltocoord_config).optimize_molecule(molecule, mol_name)
            #save_dir, final_coord = ray.get(MolToCoord.remote(self.moltocoord_config).optimize_molecule.remote(molecule, mol_name))

        if save_dir is None or final_coord is None:
            return {"molecule": molecule, "energy": float("nan"), "wavelength": float("nan"), "f_osc": float("nan")}

        # run stda and get results
        stda_save_dir = os.path.join(save_dir, 'log_stda_xtb')
        if not os.path.exists(stda_save_dir): os.makedirs(stda_save_dir)
        shutil.copy2(os.path.join(save_dir, final_coord), stda_save_dir)
        os.chdir(stda_save_dir)
        energy, wavelength, f_osc = self._get_lambda_osc(final_coord)

        os.chdir(self.log_dir)
        if self.remove_scratch:
            shutil.rmtree(save_dir, ignore_errors=True)

        log_vals = {"molecule": molecule, "energy": energy, "wavelength": wavelength, "f_osc": f_osc}
        return log_vals

    def get_score_(self, molecule, mol_name=None, save_dir=None, final_coord=None, **kwargs):
        new_dict = self.get_score(molecule=molecule, mol_name=mol_name, save_dir=save_dir, final_coord=final_coord, **kwargs)
        os.chdir(self.current_dir)
        return np.array([new_dict['energy'], new_dict['f_osc']])

    def get_scores(self, molecule_list, **kwargs):
        print(molecule_list)
        #print("Pool started")
        #with multiprocessing.Pool(processes=24) as pool:
        #    results = pool.map(self.get_score, molecule_list)
        #print("Pool ended")
        results = []
        for mol in molecule_list:
            results += [self.get_score(mol)]
        prop = np.zeros((len(molecule_list), 2))
        for i, result in enumerate(results):
            prop[i,0] = result['energy']
            prop[i,1] = result['f_osc']
        os.chdir(self.current_dir)
        return prop

    def _get_lambda_osc(self, coord_file):
        self._run_xtb4stda(coord_file)
        self._run_stda()
        energy, wavelength, fosc, _ = self._output_analysis()
        return energy, wavelength, fosc

    def _run_xtb4stda(self, coord_file):
        # coord = mol_name+'_'+str(self.ff)+'_opt_xtb_opt.xyz'
        cmd = "xtb4stda {} >& xtb4stda.out".format(coord_file)
        subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=subprocess.DEVNULL)

    def _run_stda(self):
        cmd = "{} -xtb -e {} >& stda_xtb.out".format(self.stda_command, str(self.stda_cutoff))
        subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=subprocess.DEVNULL)

        # occasionally, the energy is so high there is no tda.dat saved, rerun with higher self.stda_cutoff
        if not os.path.isfile('tda.dat'):
            cmd = "{} -xtb -e {} >& stda_xtb.out".format(self.stda_command, str(self.stda_cutoff + 5))
            subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=subprocess.DEVNULL)

    def _output_analysis(self):
        with open('tda.dat') as f:
            for l in f:
                if l.startswith(' DATXY'):
                    break
            columns = ['transition', 'energy', 'fosc', 'rot_x', 'rot_y', 'rot_z']
            df = pd.read_csv(f, header=None, names=columns, sep="\s+")
            energy, fosc, rot_x, rot_y, rot_z = df.iloc[0]['energy'], df.iloc[0]['fosc'], df.iloc[0]['rot_x'], \
            df.iloc[0]['rot_y'], df.iloc[0]['rot_z']
            wavelength = 1239.84193 / energy
            rot = [rot_x, rot_y, rot_z]
            # value = f.readline().split(r"\s+")
            # wavelength, fosc, rot = value[1], value[2], value[3:6]
            return energy, wavelength, fosc, rot

if __name__ == "__main__":
    cfg = default_stda_config
    stdaxtb = STDA_XTB(**cfg)
    print('initialized')
    smiles = ['Fc1cc(F)c(F)c(-c2c[nH]c(-c3ccc4c(c3)C3(c5ccccc5-4)c4ccccc4-c4ccc(-c5cccc6n[nH]nc56)cc43)c2)c1F',
              'N#Cc(ccc1c2ccc(N(c3ccccc3)c4ccccc4)cc2)c5c1nc(c(cccc6)c6c7c8ccc(n(c9c%10cccc9)c%11c%10[nH]c%12c%11cccc%12)c7)c8n5',
              'O=C1C(=Cn2c3ccc(-c4ccc5c6c(ccc(-c7cc8cccc9c%10cccc%11cccc(c(c7)c89)c%11%10)c46)-c4nccnc4-5)cc3c3c4sc5ccccc5c4ccc32)C(=O)c2ccccc21',
              'F[B-]1(F)n2c(cc(-c3cc4ccccn4c3)c2-c2cn3ccnc3c(-c3cc4cc5sccc5cc4s3)n2)C=C2C=CC=[N+]21',
              'C#Cc1ccc2c3cccc4cc(-n5c6ccccc6c6c7ccccc7sc65)cc(c5c(-c6nc7c8ccccc8c8ccccc8c7[nH]6)ccc1c25)c43', ]
    props = stdaxtb.get_scores(smiles)
    print(props)