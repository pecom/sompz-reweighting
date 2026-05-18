import numpy as np
from astropy.table import Table, vstack, hstack, join
from astropy.io import fits
from minisom import MiniSom
import astropy.units as u
from numpy.lib.recfunctions import structured_to_unstructured
import os, sys, gc, pickle


rng = np.random.default_rng()
ddir = '/gpfs/projects/VonDerLindenGroup/padari/som-pz'
out_dir = f'{ddir}/output/models'
suffix = ''
tomographic_bins = np.array([0, 0.4, 0.6, 0.9, 2.0])
N_pdf_bins = 101

def get_mlcat(cat, verbose=False):
    pure_filt = cat['blend_diff'] == 0

    if verbose:
        blend_frac = np.sum(~pure_filt)/len(pure_filt)
        print(f"Blend fraction: {blend_frac:0.3f}")

    pure_cat = cat[pure_filt]
    color1 = (pure_cat['g_mag'] - pure_cat['r_mag']).data
    color2 = (pure_cat['r_mag'] - pure_cat['i_mag']).data
    color3 = (pure_cat['i_mag'] - pure_cat['z_mag']).data
    color4 = (pure_cat['z_mag'] - pure_cat['y_mag']).data
    i_mag = pure_cat['i_mag'].data

    zlabel = pure_cat['redshift'].data

    nan_filt = ~np.logical_or.reduce((np.isnan(color1),
                                      np.isnan(color2),
                                      np.isnan(color3),
                                      np.isnan(color4)))
    if verbose:
        print(f"Creating catalog with {np.sum(nan_filt)} entries")

    ml_cat = np.vstack((color1[nan_filt], color2[nan_filt], color3[nan_filt],
                        color4[nan_filt], i_mag[nan_filt], zlabel[nan_filt])).T
    Nfeats = 5
    photom = ml_cat[:,:Nfeats].data
    labels = ml_cat[:,5].data
    
    return photom, labels

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
    som.train(photom, N_train, random_order=True, verbose=True)
    
    return som

def label_cell_pdfs(som, photom, zs, N_pdf_bins=N_pdf_bins):
    som_ndxs = som.win_map(photom, return_indices=True)

    N_neuron = som.get_weights().shape[0]
    som_size = N_neuron * N_neuron

    spline_bins = np.linspace(0, 5, N_pdf_bins)
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
             flat_trained_pz_pdfs, suffix=suffix, out_dir=out_dir):

    with open(f'{out_dir}/som{suffix}.pkl', 'wb') as out:
        pickle.dump(som, out)

    np.save(f'{out_dir}/flat_pdfs{suffix}.npy', flat_trained_pz_pdfs)

    with open(f'{out_dir}/cell_ndxs{suffix}.pkl', 'wb') as out:
        pickle.dump(tomographic_cell_ndxs, out)

    with open(f'{out_dir}/cell_tomo_bins{suffix}.pkl', 'wb') as out:
        pickle.dump(tomographic_ndxs, out)

    return 0

def get_cats(ndxs, ddir=ddir):
    full_cats = []
    for ndx in ndxs:
        matched_cat = Table.read(f'{ddir}/labels/matched_{ndx}.fits')
        full_cats.append(matched_cat)

    full_cat = vstack(full_cats)
    return full_cat


if __name__=="__main__":
    N_samples = 10
    load_ndxs = rng.integers(0, 10240, N_samples)

    full_cat = get_cats(load_ndxs)
    photom, redshifts = get_mlcat(full_cat, verbose=True)
    
    som = create_som(photom, 20, 5000)
    tomographic_cell_ndxs, tomographic_ndxs, flat_trained_pz_pdfs = label_cell_pdfs(som, photom, redshifts, N_pdf_bins)

    store_all(som, tomographic_cell_ndxs, tomographic_ndxs, flat_trained_pz_pdfs)

