import numpy as np
from astropy.table import Table, vstack, hstack, join
from astropy.io import fits
from minisom import MiniSom
import astropy.units as u
from numpy.lib.recfunctions import structured_to_unstructured
import os, sys, gc, pickle, yaml


rng = np.random.default_rng()
ddir = '/gpfs/projects/VonDerLindenGroup/padari/som-pz'
out_dir = f'{ddir}/output/models'

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

suffix = config['suffix']
print(f"Using {suffix=:}")
tomographic_bins = np.array(config['tomographic_bins'])
som_neurons = config['som_neurons']
N_pdf_bins = config['N_pdf_bins']
bands = config['bands']
redshift_col = config['redshift']
source = config['source']

def get_mlcat(pure_cat, band_str="{band}_mag", redshift_col='redshift', bands='grizy', verbose=False):
    ml_cat = []
    nan_filt = np.ones(len(pure_cat)).astype(bool)
    for i,b in enumerate(bands[:-1]):
        nb = bands[i+1]
        color = pure_cat[band_str.format(band=b)] - pure_cat[band_str.format(band=nb)]
        nan_filt &= ~np.isnan(color)
        ml_cat.append(color.data)
        
    ml_cat.append(pure_cat[band_str.format(band='i')].data)
    ml_cat.append(pure_cat[redshift_col].data)

    if verbose:
        print(f"Creating catalog with {np.sum(nan_filt)} entries")

    ml_cat = np.array(ml_cat)
    ml_cat = (ml_cat[:,nan_filt]).T

    Nfeats = ml_cat.shape[1] - 1
    photom = ml_cat[:,:Nfeats]
    labels = ml_cat[:,Nfeats]
    
    return photom, labels, nan_filt

def cell2ndx(n1, n2, N):
    return n1*N + n2

def ndx2cell(n, N):
    n1 = n//N
    n2 = n%N
    return n1, n2

def store_blend(blend_info, suffix=suffix, out_dir=out_dir):

    blend_numer, blend_denom, som_blend_frac = blend_info

    np.save(f'{out_dir}/blend_numer{suffix}.npy', blend_numer)
    np.save(f'{out_dir}/blend_denom{suffix}.npy', blend_denom)
    np.save(f'{out_dir}/blend_weights{suffix}.npy', som_blend_frac)

    return 0

def get_cats(ndxs, ddir=ddir, source='anacal'):
    match source:
        case 'anacal':
            full_cats = []
            for ndx in ndxs:
                matched_cat = Table.read(f'{ddir}/labels/matched_{ndx}.fits')
                full_cats.append(matched_cat)

            full_cat = vstack(full_cats)
        case 'flagship':
            full_cat = Table.read(f'{ddir}/data/flagship_blend_train.fits')
    return full_cat

def blend_som(som, full_cat, band_format):
    N_neuron = som.get_weights().shape[0]
    som_size = N_neuron * N_neuron
    
    blend_numer = np.zeros(som_size)

    full_photom, _, nan_filt = get_mlcat(full_cat, bands=bands,
                                         redshift_col=redshift_col, verbose=False,
                                         band_str=band_format)

    full_map = np.array([som.winner(fp) for fp in full_photom])
    full_map_ndxs = cell2ndx(full_map[:,0], full_map[:,1], N_neuron)
    
    blend_filt = full_cat[nan_filt]['blend_diff'] > 0

    blend_denom = np.bincount(full_map_ndxs, minlength=som_size)
    for i in range(som_size):
        blend_numer[i] = np.sum(full_map_ndxs[blend_filt] == i)
    
    som_blend_frac = np.zeros(som_size)
    som_blend_frac[blend_denom > 0] = blend_numer[blend_denom > 0]/blend_denom[blend_denom > 0]

    return blend_numer, blend_denom, som_blend_frac


if __name__=="__main__":
    # N_samples = 10
    # load_ndxs = rng.integers(0, 10240, N_samples)

    load_ndxs = np.arange(10)
    full_cat = get_cats(load_ndxs, source=source)

    match source:
        case 'anacal':
            flux_format='{band}_flux_gauss2'
            mag_format='{band}_mag'
        case 'flagship':
            flux_format='lsst_{band}'
            mag_format='lsst_mag_{band}'

    pure_filt = full_cat['blend_diff'] == 0
    blend_frac = np.sum(~pure_filt)/len(pure_filt)
    print(f"Blend fraction: {blend_frac:0.3f}")

    with open(f'{out_dir}/som{suffix}.pkl', 'rb') as out:
        som = pickle.load(out)
    
    blend_numer, blend_denom, som_blend_frac = blend_som(som, full_cat, mag_format)
    print("Created SOM blend ratio.")

    store_blend([blend_numer, blend_denom, som_blend_frac], suffix=suffix)
