import numpy as np
from astropy.table import Table, vstack, hstack, join
from astropy.io import fits
from minisom import MiniSom
from numpy.lib.recfunctions import structured_to_unstructured
import os, sys, gc, pickle


rng = np.random.default_rng()
ddir = '/gpfs/projects/VonDerLindenGroup/padari/som-pz'
out_dir = f'{ddir}/output/reweight'
model_dir = f'{ddir}/output/models'
suffix = ''
i_zp = 30
som_neuron = 32

def flux_to_mlcat(cat, verbose=False):

    color1 = -2.5*np.log10(cat[:,0]/cat[:,1])
    color2 = -2.5*np.log10(cat[:,1]/cat[:,2])
    color3 = -2.5*np.log10(cat[:,2]/cat[:,3])
    color4 = -2.5*np.log10(cat[:,3]/cat[:,4])
    i_mag = 30 - 2.5*np.log10(cat[:,2])

    nan_filt = ~np.logical_or.reduce((np.isnan(color1),
                                      np.isnan(color2),
                                      np.isnan(color3),
                                      np.isnan(color4)))
    if verbose:
        print(f"Creating catalog with {np.sum(nan_filt)} entries")

    photom = np.vstack((color1[nan_filt], color2[nan_filt], color3[nan_filt],
                        color4[nan_filt], i_mag[nan_filt])).T
    
    return photom

def cell2ndx(n1, n2, N):
    return n1*N + n2

def ndx2cell(n, N):
    n1 = n//N
    n2 = n%N
    return n1, n2

def get_cats(ndxs, ddir=ddir):
    full_cats = []
    for ndx in ndxs:
        matched_cat = Table.read(f'{ddir}/labels/matched_{ndx}.fits')
        full_cats.append(matched_cat)

    full_cat = vstack(full_cats)
    return full_cat

def load_model(suffix, model_dir=model_dir):
    with open(f'{model_dir}/som{suffix}.pkl', 'rb') as out:
        som = pickle.load(out)

    flat_trained_pz_pdfs = np.load(f'{model_dir}/flat_pdfs{suffix}.npy')

    with open(f'{model_dir}/cell_ndxs{suffix}.pkl', 'rb') as out:
        tomographic_cell_ndxs = pickle.load(out)

    with open(f'{model_dir}/cell_tomo_bins{suffix}.pkl', 'rb') as out:
        tomographic_ndxs = pickle.load(out)

    return som, tomographic_cell_ndxs, tomographic_ndxs, flat_trained_pz_pdfs

def create_blends(cat, Nblends=1000, rand_weight=False, verbose=False):
    bands = list('grizy')
    fluxes = np.vstack([cat[f'{b}_flux_gauss2'].data for b in bands]).T
    neg_flux_filt = ~(np.sum(fluxes < 0, axis=1).astype(bool))
    # zero-point = 30
    i_mag = cat['i_mag'].data[neg_flux_filt]
    good_fluxes = fluxes[neg_flux_filt]
    Ngal = np.sum(neg_flux_filt)

    blend_samples = np.zeros((Nblends, 5))
    blend_inputs = np.zeros((Nblends, 2, 5))

    nfound = 0
    while nfound < Nblends:
        if verbose:
            step_size = Nblends//10
            if nfound % step_size == 0:
                print(f"Generated {nfound} blends")

        gal_ndxs = rng.integers(0, high=Ngal, size=2)
        if np.abs(np.diff(i_mag[gal_ndxs])) > 2:
            continue

        input_flux1 = good_fluxes[gal_ndxs[0]]
        input_flux2 = good_fluxes[gal_ndxs[1]]

        if rand_weight:
            weight1, weight2 = rng.uniform(size=2)
        else:
            weight1 = 1.
            weight2 = 1.

        blended_flux = weight1*input_flux1 + weight2*input_flux2

        blend_samples[nfound] = blended_flux
        blend_inputs[nfound, 0] = input_flux1
        blend_inputs[nfound, 1] = input_flux2
        
        nfound += 1 

    return blend_samples, blend_inputs

def create_matrix(blends, input1, input2, som):
    N_neuron = som.get_weights().shape[0]

    som_size = N_neuron * N_neuron
    blend_counts_matrix = np.zeros((som_size, som_size, som_size))

    blend_map = np.array([som.winner(bp) for bp in blends])
    input1_map = np.array([som.winner(bp) for bp in input1])
    input2_map = np.array([som.winner(bp) for bp in input2])

    blend_cell_ndxs = cell2ndx(blend_map[:,0], blend_map[:,1], N_neuron)
    input1_cell_ndxs = cell2ndx(input1_map[:,0], input1_map[:,1], N_neuron)
    input2_cell_ndxs = cell2ndx(input2_map[:,0], input2_map[:,1], N_neuron)

    for i, bcx in enumerate(blend_cell_ndxs):
        blend_counts_matrix[input1_cell_ndxs[i], input2_cell_ndxs[i], bcx] += 1
        blend_counts_matrix[input2_cell_ndxs[i], input1_cell_ndxs[i], bcx] += 1
    blend_normalization = np.einsum('ijk->k', blend_counts_matrix)

    cell_weights = np.zeros((som_size, som_size))
    # cell_weights[k,i] = (2 * Number of Times cell i mapped to k)/(2 * Number of mappings to k)
    for i in range(som_size):
        if blend_normalization[i] == 0:
            cell_weights[i,i] = 1
        else:
            cell_weights[i,:] = np.sum(blend_counts_matrix[:,:,i], axis=0)/blend_normalization[i]

    return blend_counts_matrix, cell_weights


if __name__=="__main__":
    # N_samples = 10
    # load_ndxs = rng.integers(0, 10240, N_samples)
    som, tomographic_cell_ndxs, tomographic_ndxs, flat_trained_pz_pdfs = load_model(suffix)

    load_ndxs = np.arange(10)
    full_cat = get_cats(load_ndxs)
    pure_cat = full_cat[full_cat['blend_diff'] <= 0]

    blend_samples, blend_inputs = create_blends(pure_cat, Nblends=100000, verbose=True)
    print("Created Sample")

    blend_ml = flux_to_mlcat(blend_samples)
    blend_input1 = flux_to_mlcat(blend_inputs[:,0,:])
    blend_input2 = flux_to_mlcat(blend_inputs[:,1,:])

    blend_counts_matrix, cell_weights = create_matrix(blend_ml, blend_input1, blend_input2, som)
    print("Created blend matrix")

    np.save(f'{out_dir}/blend_counts_matrix{suffix}.npy', blend_counts_matrix)
    np.save(f'{out_dir}/cell_weights{suffix}.npy', cell_weights)

    np.save(f'{out_dir}/blend_sample{suffix}.npy', blend_samples)
    np.save(f'{out_dir}/blend_inputs{suffix}.npy', blend_inputs)
