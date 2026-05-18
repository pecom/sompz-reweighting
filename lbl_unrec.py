import numpy as np
import pandas as pd
from astropy.table import Table, hstack, vstack, join
import scipy.stats as stats
import os, sys
from mpi4py import MPI
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("-b", "--cbin", type=int)
# 
# args = parser.parse_args()
# cbin = int(args.cbin)

hdir = os.getenv("HOME")
pdir = os.getenv("PSCRATCH")

sys.path.insert(1, f'{hdir}/codes/friendly/friendly')
sys.path.insert(1, f'{hdir}/codes/friendly')
# ddir = '/pscratch/sd/x/xiangchl/data/simulation/2025-10-15/parquet/constant_shear_random/g1/shear02/'
ddir = '/gpfs/projects/VonDerLindenGroup/padari/anacal-output/g1/shear02_mode0/'
out_dir = '/gpfs/projects/VonDerLindenGroup/padari/som-pz/labels'

from friendly.utils import FCatalog
from friendly.matchers.kdtree import FKDTree
from friendly.pruners.mag_diff import MagDiffPruner


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

ods_cols = [b+'_ab' for b in 'ugrizy']
OneDegSq = Table.read(os.environ['CATSIM_DIR'] + '/OneDegSq.fits')


def group2table(groups):
    ndx1_name = 'kdtree_idx1'
    ndx2_name = 'kdtree_idx2'

    ndx1s = []
    ndx2s = []
    blend_diff = []
    for g in groups:
        ndx1s.append(g.idx1)
        ndx2s.append(g.idx2)
        blend_diff.append(len(g.idx2) - len(g.idx1))

    blend_table = Table(data=[ndx1s, ndx2s, blend_diff], names=[ndx1_name, ndx2_name, 'blend_diff'], dtype=[object, object, int])
    blend_table['kdtree_idx1'] = blend_table['kdtree_idx1'].reshape((len(blend_table)))
    return blend_table

def recenter_ra(sample):
    ras = sample['ra']
    big_ra = ras > 180
    new_ras = ras.copy()
    new_ras[big_ra] = ras[big_ra] - 360
    sample['new_ra'] = new_ras
    return  new_ras

def match_seed(seed, skip=True):
    if skip:
        if os.path.exists(f'{out_dir}/matched_{seed}.fits'):
            return None

    obs = Table.read(f'{ddir}/measure_multiband_{seed}.fits')
    truth = Table.read(f'{ddir}/truth_{seed}.fits')
    bands = list('grizy')
    for b in bands:
        obs[f'{b}_mag'] = 30 - 2.5*np.log10(obs[f'{b}_flux_gauss2'])
    ods_phot = OneDegSq[truth['indices']][ods_cols]
    # ods_phot = OneDegSq[truth['truth_index']][ods_cols]
    phot_truth = hstack((truth, ods_phot))

    _ = recenter_ra(obs)
    _ = recenter_ra(phot_truth)

    obs['match_ndx'] = np.arange(len(obs))
    phot_truth['match_ndx'] = np.arange(len(phot_truth))
    obs.add_index('match_ndx')
    phot_truth.add_index('match_ndx')

    phot_fc = FCatalog(phot_truth, 'match_ndx', columns=phot_truth.columns)
    full_fc = FCatalog(obs, 'match_ndx', columns=obs.columns)

    candidate_boost_factor = 1/60.**2

    lazy_kdtree = FKDTree({'search_rad': candidate_boost_factor})
    kdtree_groups, _ = lazy_kdtree(full_fc, phot_fc, {'RA1': 'new_ra', 'DEC1': 'dec',
                                                      'RA2': 'new_ra', 'DEC2': 'dec'})

    mprune = MagDiffPruner({'ground_mag_limit': 28, 'space_mag_limit': 30, 'delta_mag_limit': 3})
    pruned_groups = mprune(full_fc, phot_fc, {'ground_mag_name': 'i_mag', 'space_mag_name': 'i_ab'}, kdtree_groups)

    blend_table = group2table(pruned_groups)
    # pd_table = blend_table.to_pandas()
    # pd_table.to_parquet(f'{out_dir}/unrec_{seed}.pq')
    
    zdiffs = np.zeros(len(obs))
    lower_zs = np.zeros(len(obs))

    photo_zs = phot_truth['redshift']
    true_imags = phot_truth['i_ab']

    for i, row in enumerate(blend_table):
        if len(row['kdtree_idx2']) == 0:
            continue # Spurious detection? Unmatched for some reason, let's just skip it
        elif len(row['kdtree_idx2']) == 1:
            lower_zs[i] = photo_zs[row['kdtree_idx2'][0]]
        else:
            all_zs = photo_zs[row['kdtree_idx2']].data
            zdiffs[i] = np.max(all_zs) - np.min(all_zs)
            lower_zs[i] = np.min(all_zs)

    obs['zdiff'] = zdiffs
    obs['lower_z'] = lower_zs
    obs['blend_diff'] = blend_table['blend_diff']

    obs.write(f'{out_dir}/matched_{seed}.fits', format='fits', overwrite=True)

    return None


if __name__ == "__main__":

    send_ndxs = None
    if rank==0:
        const_ndxs = np.arange(40960)
        first_half = np.arange(20480)
        second_half = np.arange(20480, 40960)
        test_ndxs = np.arange(10240)
        # split_ndxs = np.array_split(first_half, size)
        split_ndxs = np.array_split(test_ndxs, size)
        # split_ndxs = np.arange(10)
    else:
        split_ndxs = None

    split_ndxs = comm.scatter(split_ndxs, root=0)

    print(f"Looking at {len(split_ndxs)} objects at {rank}")

    for ndx in split_ndxs:
        match_seed(ndx)

    print(f"Done with rank {rank}")
