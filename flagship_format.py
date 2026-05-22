import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from astropy.table import Table, vstack, join
import scipy.stats as stats
from matplotlib.lines import Line2D
import os, sys, gc, pickle

flagship_zp = 48.6
arcsec = 1./60.**2

# input_fname = './data/flagship_cone2.parquet'
# suffix = '_blend_train'
# only_pure = False

# input_fname = './data/flagship_cone2.parquet'
# suffix = '_train'
# only_pure = True

# input_fname = './data/flagship_cone.parquet'
# suffix = '_test'
# only_pure = False

input_fname = './data/flagship_cone2.parquet'
suffix = '_test2'
only_pure = False

# input_fname = './data/flagship_som_train.parquet'
# suffix = '_train'
# only_pure = True

def add_mags(cat):
    for b in list('ugrizy'):
        cat[f'lsst_mag_{b}'] = -2.5*np.log10(cat[f'lsst_{b}']) - flagship_zp
    for b in list('hjy'):
        cat[f'euclid_mag_{b}'] = -2.5*np.log10(cat[f'euclid_nisp_{b}']) - flagship_zp
    cat[f'euclid_mag_vis'] = -2.5*np.log10(cat[f'euclid_vis']) - flagship_zp
    return cat

def self_match(cat):
    euclid_coords = np.vstack((cat['ra_gal'], cat['dec_gal'])).T
    euclid_tree = KDTree(euclid_coords)

    match_ndxs = euclid_tree.query_ball_point(euclid_coords, 0.75*arcsec)
    return match_ndxs

def synth_cats(cat, match_ndxs):
    synth_blends = []
    synth_pure_ndxs = []

    for i,mndx in enumerate(match_ndxs):
        if len(mndx)==1:
            synth_pure_ndxs.append(mndx[0])
        else:
            if mndx[0] >= i:
                m_rows = cat[mndx]
                flux_weight = m_rows['lsst_i']
                weight_sum = np.sum(flux_weight)
                obs_ra = np.dot(m_rows['ra_gal'], flux_weight)/weight_sum
                obs_dec = np.dot(m_rows['dec_gal'], flux_weight)/weight_sum
                obs_flux = []
                for b in list('ugrizy'):
                    obs_flux.append(np.sum(m_rows[f'lsst_{b}']))
                for b in list('hjy'):
                    obs_flux.append(np.sum(m_rows[f'euclid_nisp_{b}']))
                obs_flux.append(np.sum(m_rows[f'euclid_vis']))
                redshifts = m_rows['true_redshift_gal'].data
                full_row = [obs_ra, obs_dec, redshifts, *obs_flux, mndx]
                synth_blends.append(full_row)

    synth_blend = Table(names=['ra', 'dec', 'zs',
             'lsst_u', 'lsst_g', 'lsst_r', 'lsst_i', 'lsst_z', 'lsst_y',
             'euclid_nisp_h', 'euclid_nisp_j', 'euclid_nisp_y', 'euclid_vis',
             'match_ndxs'],
      dtype=['f', 'f', 'O',
              'f', 'f', 'f', 'f', 'f', 'f',
              'f', 'f', 'f', 'f',
              'O'],
      rows=synth_blends
     )
    zdiffs = np.array([snz.max() - snz.min() for snz in synth_blend['zs']])
    lowerz = np.array([snz.min() for snz in synth_blend['zs']])

    synth_blend['zdiff'] = zdiffs
    synth_blend['lower_z'] = lowerz
    synth_blend['blend_diff'] = 1

    synth_pure = cat[synth_pure_ndxs]

    synth_pure['zdiff'] = 0
    synth_pure['lower_z'] = synth_pure['true_redshift_gal']
    synth_pure['blend_diff'] = 0

    _ = add_mags(synth_blend)
        
    synth_pure_bright = synth_pure[synth_pure['lsst_mag_i'] < 25]
    synth_blend_bright =synth_blend[synth_blend['lsst_mag_i'] < 25]


    return synth_pure_bright, synth_blend_bright

def pure_only(cat):
    cat['zdiff'] = 0
    cat['lower_z'] = cat['true_redshift_gal']
    cat['blend_diff'] = 0

    bright_cat = cat[cat['lsst_mag_i'] < 25]
    som_cols = ([f'lsst_{b}' for b in 'ugrizy'] +
            [f'lsst_mag_{b}' for b in 'ugrizy'] +
            ['lower_z', 'zdiff', 'blend_diff'])

    flag_som = bright_cat[som_cols]
    return flag_som

def som_format(pure, blend):
    som_cols = ([f'lsst_{b}' for b in 'ugrizy'] +
            [f'lsst_mag_{b}' for b in 'ugrizy'] +
            ['lower_z', 'zdiff', 'blend_diff'])

    som_pure = pure[som_cols]
    som_blend = blend[som_cols]
    flag_som = vstack((som_pure, som_blend))
    return flag_som



if __name__ == "__main__":
    test_pat = pd.read_parquet(input_fname)
    cat = Table.from_pandas(test_pat)
    print("Loaded catalog")


    _ = add_mags(cat)

    if only_pure:
        flagship = pure_only(cat)
    else:
        match_ndx = self_match(cat)
        print("Self matched")

        pure, blend = synth_cats(cat, match_ndx)
        print("Generated synthetic catalogs")
        flagship = som_format(pure, blend)

    flagship.write(f'./data/flagship{suffix}.fits', format='fits', overwrite=True)


