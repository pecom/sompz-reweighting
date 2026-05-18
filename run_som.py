import numpy as np
from astropy.table import Table, vstack, hstack, join
from astropy.io import fits
from minisom import MiniSom
import astropy.units as u
from numpy.lib.recfunctions import structured_to_unstructured
import os, sys, gc, pickle


rng = np.random.default_rng()
ddir = '/gpfs/projects/VonDerLindenGroup/padari/som-pz'
model_dir = f'{ddir}/output/models'
suffix = ''
tomographic_bins = np.array([0, 0.4, 0.6, 0.9, 2.0])
N_pdf_bins = 101

def get_photom(cat, verbose=False):
    if verbose:
        pure_filt = cat['blend_diff'] <= 0
        blend_frac = np.sum(~pure_filt)/len(pure_filt)
        print(f"Blend fraction: {blend_frac:0.3f}")

    color1 = (cat['g_mag'] - cat['r_mag']).data
    color2 = (cat['r_mag'] - cat['i_mag']).data
    color3 = (cat['i_mag'] - cat['z_mag']).data
    color4 = (cat['z_mag'] - cat['y_mag']).data
    i_mag = cat['i_mag'].data

    photom = np.vstack((color1, color2, color3,
                        color4, i_mag)).T
    
    return photom

def cell2ndx(n1, n2, N):
    return n1*N + n2

def ndx2cell(n, N):
    n1 = n//N
    n2 = n%N
    return n1, n2

def load_model(suffix, model_dir=model_dir):
    with open(f'{model_dir}/som{suffix}.pkl', 'rb') as out:
        som = pickle.load(out)

    flat_trained_pz_pdfs = np.load(f'{model_dir}/flat_pdfs{suffix}.npy')

    with open(f'{model_dir}/cell_ndxs{suffix}.pkl', 'rb') as out:
        tomographic_cell_ndxs = pickle.load(out)

    with open(f'{model_dir}/cell_tomo_bins{suffix}.pkl', 'rb') as out:
        tomographic_ndxs = pickle.load(out)

    return som, tomographic_cell_ndxs, tomographic_ndxs, flat_trained_pz_pdfs

def get_cats(ndxs, ddir=ddir):
    full_cats = []
    for ndx in ndxs:
        matched_cat = Table.read(f'{ddir}/labels/matched_{ndx}.fits')
        full_cats.append(matched_cat)

    full_cat = vstack(full_cats)
    return full_cat

def label_cells(photom, som, tomo_cell_ndxs, flat_pdfs, full_cat):
    N_neuron = som.get_weights().shape[0]
    photom_mapped = np.array([som.winner(tp) for tp in photom])
    photom_cell_ndxs = cell2ndx(photom_mapped[:,0], photom_mapped[:,1], N_neuron)

    spline_bins = np.linspace(0, 5, N_pdf_bins)

    cat_counts = np.zeros(4)
    pdfs = np.zeros((4, N_pdf_bins-1))

    true_counts = np.zeros(4)
    true_pdfs = np.zeros((4, N_pdf_bins-1))

    for i in range(1, 5):
        # 1D NDX that belong to a tomographic bin
        bin_cells = tomo_cell_ndxs[i]

        # Isolate to objects that are within the tomographic bin
        # based on 1D NDXs
        tomo_cut = np.isin(photom_cell_ndxs, bin_cells)

        # 1D NDXs of the cells that are in the tomographic bin
        cut_photom_cell_ndxs = photom_cell_ndxs[tomo_cut] 

        # Get the number of objects in this bin
        cat_counts[i-1] = len(cut_photom_cell_ndxs)

        # More explicit but slower version of below.
        # Will need to use this version when using the SOM re-weighting
        # temp_pdf = np.zeros(N_pdf_bins-1)
        # for cndx in cut_photom_cell_ndxs:
        #     pdfs[i-1,:] += 1*flat_pdfs[cndx] 

        # Add up the PDFs for the cells
        pdfs[i-1,:] =  np.sum(flat_pdfs[cut_photom_cell_ndxs], axis=0)
        
        # True information bookkeeping:
        blend_filt = full_cat['blend_diff'] > 0
        true_counts[i-1] = np.sum(tomo_cut) + np.sum(blend_filt[tomo_cut])

        true_redshifts = np.concat((full_cat['lower_z'][tomo_cut], 
                                    full_cat['lower_z'][tomo_cut & blend_filt] + full_cat['zdiff'][tomo_cut & blend_filt]))

        tpdf, _ = np.histogram(true_redshifts, bins=spline_bins, density=False)
        true_pdfs[i-1,:] += tpdf
        
    print(cat_counts)
    print(true_counts)
        
    return cat_counts, pdfs, true_counts, true_pdfs

def update_files(cat_counts, pdfs, true_cat_counts, true_pdfs):
    counts = np.load(f'./output/counts{suffix}.npy')
    counts += cat_counts
    np.save(f'./output/counts{suffix}.npy', counts)

    for i in range(4):
        tomo = np.load(f'./output/tomo{i+1}{suffix}.npy')
        tomo += pdfs[i]
        np.save(f'./output/tomo{i+1}{suffix}.npy', tomo)

    true_counts = np.load(f'./output/true_counts{suffix}.npy')
    true_counts += true_cat_counts
    np.save(f'./output/true_counts{suffix}.npy', true_counts)

    for i in range(4):
        tomo = np.load(f'./output/true_tomo{i+1}{suffix}.npy')
        tomo += true_pdfs[i]
        np.save(f'./output/true_tomo{i+1}{suffix}.npy', tomo)

    return None
    

if __name__=="__main__":
    # N_samples = 10
    # load_ndxs = rng.integers(0, 10240, N_samples)
    som, tomographic_cell_ndxs, tomographic_ndxs, flat_trained_pz_pdfs = load_model(suffix)

    for i in range(20, 30):
        load_ndxs = [i]
        full_cat = get_cats(load_ndxs)
        photom = get_photom(full_cat, verbose=True)

        cat_counts, pdfs, true_cat_counts, true_pdfs = label_cells(photom, som, tomographic_cell_ndxs, flat_trained_pz_pdfs, full_cat)

        update_files(cat_counts, pdfs, true_cat_counts, true_pdfs)


    
