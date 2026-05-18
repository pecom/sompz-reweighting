import numpy as np


N_pdf_bins = 101
suffix = ''

zero_counts = np.zeros(4)
zero_bin = np.zeros(N_pdf_bins-1)

np.save(f'./output/counts{suffix}.npy', zero_counts)
np.save(f'./output/tomo1{suffix}.npy', zero_bin)
np.save(f'./output/tomo2{suffix}.npy', zero_bin)
np.save(f'./output/tomo3{suffix}.npy', zero_bin)
np.save(f'./output/tomo4{suffix}.npy', zero_bin)

np.save(f'./output/true_counts{suffix}.npy', zero_counts)
np.save(f'./output/true_tomo1{suffix}.npy', zero_bin)
np.save(f'./output/true_tomo2{suffix}.npy', zero_bin)
np.save(f'./output/true_tomo3{suffix}.npy', zero_bin)
np.save(f'./output/true_tomo4{suffix}.npy', zero_bin)
