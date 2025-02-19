import random, subprocess, re, shutil, warnings, traceback
import os, time
from rdkit import Chem
from rdkit.Chem import AllChem
from utils.utils import natural_sort, get_files_in_dir
import numpy as np

# to fix shutil.Error: Destination path already exists


def get_conformers(mol, num_conf=500, maxattempts=100, randomcoords=True, prunermsthres=-1, return_mol=False,
                   filename=None, filepath=None, save_individual=True, **kwargs):
    if type(mol) is str:
        mol = Chem.MolFromSmiles(mol)
    Chem.SanitizeMol(mol)
    mol_h = Chem.AddHs(mol)
    # print('done_pre_embedding')
    AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conf, maxAttempts=maxattempts,
                               pruneRmsThresh=prunermsthres, useRandomCoords=randomcoords)
    # print('done_embedding')
    if mol_h.GetNumConformers() == 0:
        # for difficult to embed conformers
        AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conf, maxAttempts=1000, clearConfs=True,
                                   pruneRmsThresh=prunermsthres, useRandomCoords=False)
    if return_mol: return mol_h
    if filepath is not None:
        if not os.path.exists(filepath): 
            os.makedirs(filepath)
        if filename is None:
            filename = str(random.randint(1, int(1e6)))
        if save_individual:
            for cid in range(mol_h.GetNumConformers()):
                Chem.MolToXYZFile(mol_h, os.path.join(filepath, filename+'_conf_'+str(cid)+'.xyz'), confId=int(cid))
                print(str(cid)+'th conformer saved')
        else:
            writer = Chem.SDWriter(os.path.join(filepath, filename))
            for cid in range(mol.GetNumConformers()):
                writer.write(mol, confId=cid)
    else: 
        raise NotImplementedError

def get_rdkit_ff_coordinates(mol, FF='MMFF', filename=None, filepath=None, save_min_energy=False,
                             xyzblock=False, return_mol=False, conformer_config=None,
                             **kwargs):
    '''
    Takes an input molecule and convert to optimized 3D coordinates, can also save coordinates and call to write Orca/Gaussian input files

    :param mol: molecule in Mol format, or SMILES string
    :param FF: Molecular mechanics forcefiled
    :param num_conf: number of configurations to generate, 50 for num_rot_bonds < 7; 200 for 8 <= num_rot_bonds <= 12; 300 otherwise
    :param filename: saves .xyz file with this filename
    :param filepath: ^ but filepath
    :param maxattempts: max attempts at embedding conformer
    :param randomcoords: whether to use random coordinates
    :param prunermsthres: whether to use a RMSD threshold to keep some conformers only
    :param xyzblock: returns xyzblock
    :param return_mol: returns molecule
    '''
    start_time = time.time()
    if type(mol) is str:
        mol = Chem.MolFromSmiles(mol)
    Chem.SanitizeMol(mol)
    mol_h = Chem.AddHs(mol)

    if conformer_config is None:
        conformer_config = {"num_conf": 4, "maxattempts":100, "randomcoords": True, "prunermsthres": 1}
    num_conf = conformer_config["num_conf"]
    prunermsthres = conformer_config["prunermsthres"]
    randomcoords = conformer_config["randomcoords"]
    maxattempts = conformer_config["maxattempts"]

    AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conf, pruneRmsThresh=prunermsthres, maxAttempts=maxattempts, useRandomCoords=randomcoords) # prunermsthres appear to cause issues later
    num_conf = mol_h.GetNumConformers() # get new number after pruning
    conformer_generation_time = time.time() - start_time
    if FF.lower() == 'mmff':
        try:
            msg = [AllChem.MMFFOptimizeMolecule(mol_h, confId=i, maxIters=1000) for i in range(num_conf)]
            # print(msg, Chem.MolToSmiles(mol))
        except Exception as e: print(e)
        mp = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant='MMFF94')
        mi = np.argmin(
            [AllChem.MMFFGetMoleculeForceField(mol_h, mp, confId=i).CalcEnergy() for i in range(num_conf)])
    elif FF.lower() == 'uff':
        try:
            msg = [AllChem.UFFOptimizeMolecule(mol_h, confId=i, maxIters=1000) for i in range(num_conf)]
            # print(msg, Chem.MolToSmiles(mol))
        except Exception as e: print(e)
        mi = np.argmin([AllChem.UFFGetMoleculeForceField(mol_h, confId=i).CalcEnergy() for i in range(num_conf)])
    else:
        raise NotImplementedError
    xyz_file = None
    if filepath is not None and filename is not None:
        if not os.path.exists(filepath): os.makedirs(filepath)
        xyz_file = os.path.join(filepath, str(filename) + '_' + FF + '_opt.xyz')
        Chem.MolToXYZFile(mol_h, xyz_file, confId=int(mi))  # save xyz
        if save_min_energy:
            min_energy = np.min([AllChem.UFFGetMoleculeForceField(mol_h, confId=i).CalcEnergy() for i in range(num_conf)])
            print('minimum energy is '+str(min_energy))
            with open(os.path.join(filepath, str(filename) + '_' + FF + '_opt_energy.txt'), "w") as f:
                f.write('total energy \n'+str(min_energy))
                f.write('\n conformer generation time \n' + str(conformer_generation_time))
                f.write('\n total optimization time \n')
                f.write(str(time.time()-start_time))
                f.close()
    if xyzblock:
        return Chem.MolToXYZBlock(mol_h, confId=int(mi))  # return xyz
    if return_mol:
        return mol_h, int(mi)#.GetConformer(id=int(mi)).GetOwningMol() # return mol
    print('total FF optimization time is', time.time()-start_time)
    return xyz_file # return xyz_file name


def run_gfn_ff(filepath, mols=None, opt=True, filename=str(random.randint(1, 10000000000000)), gfn_ff_config:str="", conformer_config:dict={}, coord_file_format='xyz',
               get_lowest_conformer=True, lowest_conformer_dir="", log_file_dir="log_gfn_ff", conformer_file_suffix='_conf_',
               **kwargs):
    '''
    Runs GFN_FF given a directory and either it contains coord files or a list of smiles/mols are supplied.

    :param filepath: directory for which xyz files are contained, where log and new xyz files are saved
    :param mols: an optional [list of] smiles or Mol for which conformers will be generated (default is None)
    :param opt: FF optimization or SP (default is opt)
    :param gfn_ff_config: additional GFN_FF config (default is None)
    :param conformer_config: additional conformer generation config (default is None)
    :param coord_file_format: coordinate file file format (default is xyz)
    :param get_lowest_conformer: for each molecule with numerous conformers, find the lowest energy conformer by FF (default is True)
    :param conformer_file_suffix: use this as a separator to indicate the molecule (in separation of the conformer) (default is _conf_, used in get_conformers)
    :param lowest_conformer_dir: for each molecule with numerous conformers, save the lowest energy conformer here (default if filepath, i.e. "")
    :param log_file_dir: redirect log file to another directory (default is filepath)
    '''
    if not os.path.exists(filepath): os.makedirs(filepath)
    if mols:
        # conformer_config['filepath'] = filepath
        if type(mols) is list:
            for i, mol in enumerate(mols):
                print('generating conformers for molecule number {}'.format(i))
                # conformer_config['filename'] = str(i)
                get_conformers(mol, filepath=filepath, filename=str(i), **conformer_config)
        else:
            # conformer_config['filename'] = filename
            print('generating conformers for {}'.format(filename))
            get_conformers(mols, filepath=filepath, filename=filename, **conformer_config)

    xyz_files = get_files_in_dir(filepath, "*."+coord_file_format)
    if opt: opt = "--opt"
    else: opt=""

    xyz_files=[w.split('/')[-1] for w in xyz_files]
    lowest_conformer_dir = os.path.join(filepath, lowest_conformer_dir)
    if len(lowest_conformer_dir)>0 and not os.path.exists(lowest_conformer_dir):
        os.makedirs(lowest_conformer_dir) # make directory to save lowest_conformer

    log_file_dir = os.path.join(filepath, log_file_dir)
    if len(log_file_dir)>0 and not os.path.exists(log_file_dir):
        os.makedirs(log_file_dir) # make directory to move all log_files
    starting_dir = os.getcwd()
    os.chdir(filepath)
    lowest_energy = 0
    lowest_mol = None
    print(xyz_files)
    for xyz_file in xyz_files:
        print(xyz_file)
        file_name = xyz_file.split('.')[0]
        cmd = "xtb --gfnff {} {} {} 2>&1 | tee -a {}".format(xyz_file, opt, gfn_ff_config, file_name+'.out') # xtb is weird in output redirection
        subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=subprocess.DEVNULL)
        shutil.move(xyz_file, os.path.join(log_file_dir, os.path.basename(xyz_file)))
        shutil.move(os.path.join(filepath, file_name+'.out'), os.path.join(log_file_dir, file_name+'.out'))
        if os.path.isfile(os.path.join(filepath, 'NOT_CONVERGED')):
            # if 'abnormal termination' in open(os.path.join(log_file_dir, file_name) + '.out').read():
            warnings.warn('abnormal termination for {}'.format(file_name))
            shutil.move(os.path.join(filepath, 'xtblast.xyz'), os.path.join(log_file_dir, file_name + '_xtblast_failed_gfnff_opt.xyz'))
            shutil.move(os.path.join(filepath, 'NOT_CONVERGED'), os.path.join(log_file_dir, file_name + '_NOT_CONVERGED'))
            continue
        #if not os.path.exists(os.path.join(filepath, 'xtbopt.xyz')):
        shutil.move(os.path.join(filepath, 'xtbopt.xyz'), os.path.join(log_file_dir, file_name + '_gfnff_opt.xyz'))
        # some versions of xtb do not produce these files
        if os.path.exists(os.path.join(filepath, 'xtbopt.log')):
            shutil.move(os.path.join(filepath, 'xtbopt.log'), os.path.join(log_file_dir, file_name + '_gfnff_opt.log'))
        if os.path.exists(os.path.join(filepath, 'gfnff_charges')):
            shutil.move(os.path.join(filepath, 'gfnff_charges'), os.path.join(log_file_dir, file_name + '_gfnff_charges'))
        if os.path.exists(os.path.join(filepath, 'gfnff_adjacency')):
            shutil.move(os.path.join(filepath, 'gfnff_adjacency'), os.path.join(log_file_dir, file_name + '_gfnff_adjacency'))
        # do not reuse topology for now
        if os.path.exists(os.path.join(filepath, 'gfnff_topo')):
            shutil.move(os.path.join(filepath, 'gfnff_topo'), os.path.join(log_file_dir, 'gfnff_topo'))
        if get_lowest_conformer:
            # find lowest energy
            with open(os.path.join(log_file_dir, file_name + '.out'), 'r') as f:
                for line in f.readlines():
                    if 'TOTAL ENERGY' in line:
                        energy = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
                        break
            print('current energy for {} is at {}'.format(xyz_file, energy))

            mol_file_prefix = xyz_files[0].split(conformer_file_suffix)[0]
            current_mol_file_prefix = file_name.split(conformer_file_suffix)[0] # check if it's a new molecule or same but different conformer
            if current_mol_file_prefix != mol_file_prefix: # for a new molecule,  print lowest energy, reset lowest energy & prefix,
                print('lowest energy for {} is {} at {}'.format(mol_file_prefix, xyz_file, energy))
                lowest_energy=0
                mol_file_prefix = current_mol_file_prefix

            if energy <= lowest_energy:
                lowest_energy = energy
                lowest_mol = xyz_file
                print('current lowest for {} is {} at {}'.format(mol_file_prefix, xyz_file, energy))
                shutil.copy2(os.path.join(log_file_dir, file_name+'_gfnff_opt.xyz'), os.path.join(lowest_conformer_dir, mol_file_prefix+'_lowest_energy_gfnffopt.xyz'))
    os.chdir(starting_dir)

def run_gfn_xtb(filepath, filename=None, gfn_version='2', opt=True, gfn_xtb_config:str=None, coord_file_format='xyz',
                optimized_xyz_dir=None, log_file_dir="log_gfn_xtb", **kwargs):
    '''
    Runs GFN_FF given a directory and either a coord file or all coord files will be run

    :param filepath: Directory containing the coord file
    :param filename: if given, the specific coord file to run
    :param gfn_version: GFN_xtb version (default is 2)
    :param opt: optimization or singlet point (default is opt)
    :param gfn_xtb_config: additional xtb config (default is None)
    :param coord_file_format: coordinate file format if all coord files in filepath is run (default is xyz)
    :return:
    '''
    if filename is None:
        xyz_files = get_files_in_dir(filepath, "*."+coord_file_format)
        xyz_files = [w.split('/')[-1] for w in xyz_files]
    else:
        xyz_files = [os.path.join(filepath, filename)]

    if opt: opt = "--opt"
    else: opt=""
    starting_dir = os.getcwd()
    os.chdir(filepath)
    log_file_dir = os.path.join(filepath, log_file_dir)
    if not os.path.exists(log_file_dir): os.makedirs(log_file_dir)
    for xyz_file in xyz_files:
        file_name = str(xyz_file.split('.')[0])
        cmd = "xtb --gfn {} {} {} {} 2>&1 | tee -a {}".format(str(gfn_version), xyz_file, opt, gfn_xtb_config, file_name+'.out')  # xtb is weird in output redirection
        subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=subprocess.DEVNULL)
        if os.path.isfile(os.path.join(filepath, 'NOT_CONVERGED')):
            # todo alternatively try gfn0-xtb and then gfn2-xtb
            warnings.warn('xtb --gfn {} for {} is not converged, using last optimized step instead; proceed with caution'.format(str(gfn_version), file_name))
            shutil.move(os.path.join(filepath, 'xtblast.xyz'), os.path.join(log_file_dir, os.path.basename(file_name)+'_xtb_opt.xyz'))
            shutil.move(os.path.join(filepath, 'NOT_CONVERGED'), os.path.join(log_file_dir, 'NOT_CONVERGED'))
        elif not os.path.isfile(os.path.join(filepath, 'xtbopt.xyz')): #other abnormal convergence:
            warnings.warn('xtb --gfn {} for {} abnormal termination, likely scf issues, using initial geometry instead; proceed with caution'.format(str(gfn_version), file_name))
            shutil.copy2(xyz_file, os.path.join(log_file_dir, os.path.basename(file_name)+'_xtb_opt.xyz'))
        else:
            shutil.move(os.path.join(filepath, 'xtbopt.xyz'), os.path.join(log_file_dir, os.path.basename(file_name)+'_xtb_opt.xyz'))
            try:
                shutil.move(os.path.join(filepath, 'xtbopt.log'), os.path.join(log_file_dir, 'xtbopt.log'))
                shutil.move(os.path.join(filepath, 'xtbtopo.mol'), os.path.join(log_file_dir, 'xtbtopo.mol'))
                shutil.move(os.path.join(filepath, 'wbo'), os.path.join(log_file_dir, 'wbo'))
                shutil.move(os.path.join(filepath, 'charges'), os.path.join(log_file_dir, 'charges'))
                shutil.move(os.path.join(filepath, 'xtbrestart'), os.path.join(log_file_dir, 'restart'))
            except Exception:
                # some versions of xtb do not produce these files
                print('Traceback run_gfn_xtb')
                traceback.print_exc()
                pass
            print('{} xtb optimization is done'.format(file_name))
        shutil.move(file_name+'.out', log_file_dir)
        if optimized_xyz_dir:
            if not os.path.exists(optimized_xyz_dir): os.makedirs(optimized_xyz_dir)  # make directory to save xyz file
            shutil.copy2(os.path.join(log_file_dir, os.path.basename(file_name)+'_xtb_opt.xyz'), os.path.join(optimized_xyz_dir, os.path.basename(file_name).split('.')[0] + '_xtb_opt.xyz'))
    os.chdir(starting_dir)

# run_gfn_ff("../../Admin/Pre-optimization/all_conformers/", get_lowest_conformer=True, lowest_conformer_dir="../..//Admin/Pre-optimization/lowest_conformers/")
# run_gfn_xtb('../../Admin//Pre-optimization/test/')