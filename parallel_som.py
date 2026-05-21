import numpy as np
from astropy.table import Table, vstack, hstack, join
from astropy.io import fits
from minisom import MiniSom
import astropy.units as u
from numpy.lib.recfunctions import structured_to_unstructured
from mpi4py import MPI
import os, sys, gc, pickle, yaml

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
bands = config['bands']

def get_photom(cat, band_str='{band}_mag', bands='grizy', verbose=False):
    if verbose:
        pure_filt = cat['blend_diff'] <= 0
        blend_frac = np.sum(~pure_filt)/len(pure_filt)
        print(f"Blend fraction: {blend_frac:0.3f}")

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

def get_cats(ndxs, ddir=ddir):
    full_cats = []
    for ndx in ndxs:
        matched_cat = Table.read(f'{ddir}/labels/matched_{ndx}.fits')
        full_cats.append(matched_cat)

    full_cat = vstack(full_cats)
    return full_cat

def label_cells(photom, som, tomo_cell_ndxs,
                flat_pdfs, full_cat, blend_weights, cell_weights):
    N_neuron = som.get_weights().shape[0]
    photom_mapped = np.array([som.winner(tp) for tp in photom])
    photom_cell_ndxs = cell2ndx(photom_mapped[:,0], photom_mapped[:,1], N_neuron)

    spline_bins = np.linspace(0, 5, N_pdf_bins)

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

def update_files(cat_counts, base_pdfs, weight_pdfs, true_cat_counts, true_pdfs, suffix=suffix):
    add_file(cat_counts, f'./output/counts{suffix}.npy')
    add_file(base_pdfs, f'./output/base_pdfs{suffix}.npy')
    add_file(weight_pdfs, f'./output/weight_pdfs{suffix}.npy')
    add_file(true_cat_counts, f'./output/true_counts{suffix}.npy')
    add_file(true_pdfs, f'./output/true_pdfs{suffix}.npy')

    return None
    

if __name__=="__main__":


    som_full_size = som_neurons * som_neurons

    if rank == 0:
        # full_ndxs = np.arange(10, 10240)
        # full_ndxs = np.arange(210, 310)
        full_ndxs = np.arange(10, 30)
        load_ndxs = np.array_split(full_ndxs, size)
        som, tomographic_cell_ndxs, tomographic_ndxs, flat_trained_pz_pdfs = load_model(suffix)

        cell_weights = np.load(f'./output/reweight/cell_weights.npy')
        blend_weights = np.load(f'./output/models/blend_weights.npy')
    else:
        load_ndxs = None
        som = None
        tomographic_cell_ndxs = None
        tomographic_ndxs = None
        flat_trained_pz_pdfs = np.empty((som_full_size, N_pdf_bins-1), dtype='float64')
        cell_weights = np.empty((som_full_size,som_full_size), dtype='float64')
        blend_weights = np.empty(som_full_size, dtype='float64')

    load_ndxs = comm.scatter(load_ndxs, root=0)
    som = comm.bcast(som, root=0)
    tomographic_cell_ndxs = comm.bcast(tomographic_cell_ndxs, root=0)
    tomographic_ndxs = comm.bcast(tomographic_ndxs, root=0)
    comm.Bcast(cell_weights, root=0)
    comm.Bcast(flat_trained_pz_pdfs, root=0)
    comm.Bcast(blend_weights, root=0)


    print(rank, size, load_ndxs)

    real_counts = np.zeros(4)
    true_counts = np.zeros(4)
    base_pdfs = np.zeros((4, N_pdf_bins - 1))
    weight_pdfs = np.zeros((4, N_pdf_bins - 1))
    true_pdfs = np.zeros((4, N_pdf_bins - 1))

    comm.Barrier()

    for i in load_ndxs:
        single_ndx = [i]
        full_cat = get_cats(single_ndx)
        photom = get_photom(full_cat, verbose=False)

        cat_counts, bpdfs, wpdfs, true_cat_counts, tpdfs = label_cells(photom, som, tomographic_cell_ndxs,
                                                                       flat_trained_pz_pdfs, full_cat,
                                                                       blend_weights, cell_weights)
        real_counts += cat_counts
        true_counts += true_cat_counts

        base_pdfs += bpdfs
        weight_pdfs += wpdfs
        true_pdfs += tpdfs

        print(f"Finished with ndx {i} on {rank}")


    recv_counts = None
    recv_tcounts = None
    recv_bpdfs = None
    recv_wpdfs = None
    recv_tpdfs = None

    if rank == 0:
        recv_counts = np.empty([size, 4], dtype='float64')
        recv_tcounts = np.empty([size, 4], dtype='float64')
        recv_bpdfs = np.empty([size, 4, N_pdf_bins - 1], dtype='float64')
        recv_wpdfs = np.empty([size, 4, N_pdf_bins - 1], dtype='float64')
        recv_tpdfs = np.empty([size, 4, N_pdf_bins - 1], dtype='float64')

    comm.Gather(real_counts, recv_counts, root=0)
    comm.Gather(true_counts, recv_tcounts, root=0)
    comm.Gather(base_pdfs, recv_bpdfs, root=0)
    comm.Gather(weight_pdfs, recv_wpdfs, root=0)
    comm.Gather(true_pdfs, recv_tpdfs, root=0)
    

    if rank==0:
        update_files(np.sum(recv_counts, axis=0), np.sum(recv_bpdfs, axis=0),
                    np.sum(recv_wpdfs, axis=0), np.sum(recv_tcounts, axis=0),
                    np.sum(recv_tpdfs, axis=0))
        # update_files(cat_counts, base_pdfs, weight_pdfs, true_cat_counts, true_pdfs)

