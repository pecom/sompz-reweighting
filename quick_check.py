import numpy as np

N_pdf_bins = 101
suffix = ''


counts = np.load(f'./output/counts{suffix}.npy')
tcounts = np.load(f'./output/true_counts{suffix}.npy')

print(counts)
print(tcounts)

# np.save(f'./output/base_pdfs{suffix}.npy', zero_bin)
# np.save(f'./output/weight_pdfs{suffix}.npy', zero_bin)
# np.save(f'./output/true_pdfs{suffix}.npy', zero_bin)

# for i in range(1,5):
#     np.save(f'./output/base_tomo{i}{suffix}.npy', zero_bin)
#     np.save(f'./output/weight_tomo{i}{suffix}.npy', zero_bin)
#     np.save(f'./output/true_tomo{i}{suffix}.npy', zero_bin)

