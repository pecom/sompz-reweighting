import numpy as np
from astropy.table import Table, vstack, hstack, join
from astropy.io import fits
from minisom import MiniSom
import astropy.units as u
from numpy.lib.recfunctions import structured_to_unstructured
import os, sys, gc, pickle, yaml
import argparse


rng = np.random.default_rng()
ddir = '/gpfs/projects/VonDerLindenGroup/padari/som-pz'
model_dir = f'{ddir}/output/models'

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

suffix = config['suffix']
print(f"Using {suffix=:}")
tomographic_bins = np.array(config['tomographic_bins'])
som_neurons = config['som_neurons']
N_pdf_bins = config['N_pdf_bins']
spline_max = config['spline_max']
bands = config['bands']
source = config['source']

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", default="", type=str,
                    help="Input suffix to ./data/flagship_{INPUT}")
parser.add_argument("-o", "--output", default="", type=str,
                    help="Output suffix to ./output/{OUTPUT}_pdf{suffix}.npy")
parser.add_argument("-r", "--reweight", action='store_true',
                    help="Use SOM Reweighting")
args = parser.parse_args()
input_suffix = args.input
out_suffix = args.output
reweight = args.reweight

def get_photom(cat, band_str="{band}_mag", bands='grizy', verbose=False):
    if verbose:
        pure_filt = cat['blend_diff'] <= 0
        blend_frac = np.sum(~pure_filt)/len(pure_filt)
        print(f"Blend fraction: {blend_frac:0.3f}")

    # color1 = (cat['g_mag'] - cat['r_mag']).data
    # color2 = (cat['r_mag'] - cat['i_mag']).data
    # color3 = (cat['i_mag'] - cat['z_mag']).data
    # color4 = (cat['z_mag'] - cat['y_mag']).data
    # i_mag = cat['i_mag'].data

    # photom = np.vstack((color1, color2, color3,
    #                     color4, i_mag)).T
    
    photom = []
    nan_filt = np.ones(len(cat)).astype(bool)
    for i,b in enumerate(bands[:-1]):
        nb = bands[i+1]
        color = cat[band_str.format(band=b)] - cat[band_str.format(band=nb)]
        nan_filt &= ~np.isnan(color)
        photom.append(color.data)
        
    photom.append(cat[band_str.format(band='i')].data)

    photom = np.array(photom).T

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

def get_cats(ndxs, ddir=ddir, source='anacal'):
    match source:
        case 'anacal':
            full_cats = []
            for ndx in ndxs:
                matched_cat = Table.read(f'{ddir}/labels/matched_{ndx}.fits')
                full_cats.append(matched_cat)

            full_cat = vstack(full_cats)
        case 'flagship':
            full_cat = Table.read(f'{ddir}/data/flagship_{input_suffix}.fits')
    return full_cat


def label_cells(photom, som, tomo_ndxs, flat_pdfs,
                reweight=False, blend_weights=None, cell_weights=None):

    N_neuron = som.get_weights().shape[0]
    photom_mapped = np.array([som.winner(tp) for tp in photom])
    photom_cell_ndxs = cell2ndx(photom_mapped[:,0], photom_mapped[:,1], N_neuron)

    spline_bins = np.linspace(0, spline_max, N_pdf_bins)
    base_pdfs = np.zeros((5, N_pdf_bins-1))
    debug_counts = np.zeros(5)

    # Not the optimized way to do this but it is easier to read
    # and easier to code :)
    for i,pm in enumerate(photom_mapped):
        gal_bin = tomo_ndxs[(pm[0], pm[1])]
        if gal_bin >= 6:
            continue
        cndx = photom_cell_ndxs[i]
        debug_counts[gal_bin-1] += 1
        if reweight:
            pure_weight = 1 - blend_weights[cndx]
            blend_weight = blend_weights[cndx]
            base_pdfs[gal_bin-1,:] += (pure_weight * flat_pdfs[cndx] +
                                       blend_weight * np.dot(cell_weights[cndx,:], flat_pdfs))
        else:
            base_pdfs[gal_bin-1,:] += flat_pdfs[cndx]

    print(debug_counts)
    return base_pdfs


def old_label_cells(photom, som, tomo_cell_ndxs,
                flat_pdfs, full_cat, blend_weights, cell_weights):
    N_neuron = som.get_weights().shape[0]
    photom_mapped = np.array([som.winner(tp) for tp in photom])
    photom_cell_ndxs = cell2ndx(photom_mapped[:,0], photom_mapped[:,1], N_neuron)

    spline_bins = np.linspace(0, spline_max, N_pdf_bins)

    cat_counts = np.zeros(4)
    base_pdfs = np.zeros((4, N_pdf_bins-1))
    weight_pdfs = np.zeros((4, N_pdf_bins-1))

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


        # Add up the PDFs for the cells (with no weighting)
        base_pdfs[i-1,:] =  np.sum(flat_pdfs[cut_photom_cell_ndxs], axis=0)

        # Iterate over each object to get the proper weight 
        temp_pdf = np.zeros(N_pdf_bins-1)
        for cndx in cut_photom_cell_ndxs:
            pure_weight = 1 - blend_weights[cndx]
            blend_weight = blend_weights[cndx]

            weight_pdfs[i-1,:] += (pure_weight * flat_pdfs[cndx] +
                                   blend_weight * np.dot(cell_weights[cndx,:], flat_pdfs))
        
        # True information bookkeeping:
        blend_filt = full_cat['blend_diff'] > 0
        true_counts[i-1] = np.sum(tomo_cut) + np.sum(blend_filt[tomo_cut])

        true_redshifts = np.concat((full_cat['lower_z'][tomo_cut], 
                                    full_cat['lower_z'][tomo_cut & blend_filt] + full_cat['zdiff'][tomo_cut & blend_filt]))

        tpdf, _ = np.histogram(true_redshifts, bins=spline_bins, density=False)
        true_pdfs[i-1,:] += tpdf
        
    return cat_counts, base_pdfs, weight_pdfs, true_counts, true_pdfs


def add_file(add, fname):
    tmp = np.load(fname)
    tmp += add
    np.save(fname, tmp)

if __name__=="__main__":
    som, tomographic_cell_ndxs, tomographic_ndxs, flat_trained_pz_pdfs = load_model(suffix)

    cell_weights = np.load(f'./output/reweight/cell_weights.npy')
    blend_weights = np.load(f'./output/models/blend_weights.npy')

    base_pdfs = np.zeros((4, N_pdf_bins - 1))

    load_ndxs = np.arange(10, 30)
    full_cat = get_cats(load_ndxs, source=source)

    print(f"Catalog loaded with {np.sum(full_cat['blend_diff'] >= 1)/len(full_cat):0.3f} blends")

    match source:
        case 'anacal':
            flux_format='{band}_flux_gauss2'
            mag_format='{band}_mag'
        case 'flagship':
            flux_format='lsst_{band}'
            mag_format='lsst_mag_{band}'

    photom = get_photom(full_cat, verbose=False, bands=bands,
                        band_str=mag_format)

    print("Sample photom:", photom[-3:])

    bpdf = label_cells(photom, som, tomographic_ndxs, flat_trained_pz_pdfs,
                       reweight, blend_weights, cell_weights)

    add_file(bpdf, f'./output/{out_suffix}_pdfs{suffix}.npy')
