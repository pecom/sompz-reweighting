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
spline_max = config['spline_max']
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

def create_som(photom, N, N_train=1000, sigma=np.pi, learning_rate=0.5):
    N_neuron = N
    som_size = N_neuron * N_neuron
    Nfeats = photom.shape[1]
    som = MiniSom(N_neuron, N_neuron, Nfeats, sigma=sigma, learning_rate=learning_rate)
    # som.train_batch_offline_fast(photom, N_train, verbose=True)
    som.train(photom, N_train, random_order=True, verbose=True)
    
    return som

def label_cell_pdfs(som, photom, zs, N_pdf_bins=N_pdf_bins):
    som_ndxs = som.win_map(photom, return_indices=True)

    N_neuron = som.get_weights().shape[0]
    som_size = N_neuron * N_neuron

    spline_bins = np.linspace(0, spline_max, N_pdf_bins)
    zero_pdf = np.zeros(N_pdf_bins-1)
    flat_trained_pz_pdfs = np.zeros((som_size, N_pdf_bins-1))
    tomographic_ndxs = {}

    for i in range(N_neuron):
        for j in range(N_neuron):
            # Get redshifts of objects that belong in this cell
            pz_neuron_sample = zs[som_ndxs[i,j]]

            # If we have no objects, return a 0 PDF
            if len(pz_neuron_sample) == 0:
                tomographic_ndxs[i,j] = 0
                continue

            # Assign each cell to a bin based on the median of the redshifts
            tomographic_ndxs[i,j] = np.digitize(np.median(pz_neuron_sample), tomographic_bins)

            # Go from a collection of point estimates to a PDF 
            counts, _ = np.histogram(pz_neuron_sample, bins=spline_bins, density=True)
            flat_ndx = cell2ndx(i, j, N_neuron)
            flat_trained_pz_pdfs[flat_ndx] = counts

    tomographic_cell_ndxs = {i:[] for i in range(1, 6)}

    for i in range(N_neuron):
        for j in range(N_neuron):
            ndx = tomographic_ndxs[i,j]
            if ndx==0:
                continue
            cell_ndx = cell2ndx(i,j, N_neuron)
            tomographic_cell_ndxs[ndx].append(cell_ndx)
    
    return tomographic_cell_ndxs, tomographic_ndxs, flat_trained_pz_pdfs


def store_all(som, tomographic_cell_ndxs, tomographic_ndxs,
             flat_trained_pz_pdfs, blend_info, suffix=suffix, out_dir=out_dir):

    with open(f'{out_dir}/som{suffix}.pkl', 'wb') as out:
        pickle.dump(som, out)

    np.save(f'{out_dir}/flat_pdfs{suffix}.npy', flat_trained_pz_pdfs)

    with open(f'{out_dir}/cell_ndxs{suffix}.pkl', 'wb') as out:
        pickle.dump(tomographic_cell_ndxs, out)

    with open(f'{out_dir}/cell_tomo_bins{suffix}.pkl', 'wb') as out:
        pickle.dump(tomographic_ndxs, out)

    # np.save(f'{out_dir}/blend_numer{suffix}.npy', blend_numer)
    # np.save(f'{out_dir}/blend_denom{suffix}.npy', blend_denom)
    # np.save(f'{out_dir}/blend_weights{suffix}.npy', som_blend_frac)

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
            full_cat = Table.read(f'{ddir}/data/flagship_train.fits')
    return full_cat


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

    photom, redshifts, _ = get_mlcat(full_cat[pure_filt], bands=bands,
                                     redshift_col=redshift_col, verbose=True, 
                                     band_str=mag_format)
    
    print("Sample photom:", photom[:3])
    som = create_som(photom, som_neurons, 100000)
    print("Created SOM.")
    tomographic_cell_ndxs, tomographic_ndxs, flat_trained_pz_pdfs = label_cell_pdfs(som, photom, redshifts, N_pdf_bins)
    print("Assigned PZ PDFs.")

    # blend_numer, blend_denom, som_blend_frac = blend_som(som, full_cat, mag_format)
    # print("Created SOM blend ratio.")

    # store_all(som, tomographic_cell_ndxs, tomographic_ndxs, flat_trained_pz_pdfs, [blend_numer, blend_denom, som_blend_frac])
    store_all(som, tomographic_cell_ndxs, tomographic_ndxs, flat_trained_pz_pdfs, [1,2,3])
