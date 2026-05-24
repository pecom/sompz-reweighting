import numpy as np
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

suffix = config['suffix']
print(f"Using {suffix=:}")
N_pdf_bins = config['N_pdf_bins']


# zero_counts = np.zeros(4)
zero_bin = np.zeros((5, N_pdf_bins-1))

# np.save(f'./output/counts{suffix}.npy', zero_counts)
# np.save(f'./output/true_counts{suffix}.npy', zero_counts)

np.save(f'./output/obs_pdfs{suffix}.npy', zero_bin)
np.save(f'./output/weight_pdfs{suffix}.npy', zero_bin)
np.save(f'./output/best_pdfs{suffix}.npy', zero_bin)

# for i in range(1,5):
#     np.save(f'./output/base_tomo{i}{suffix}.npy', zero_bin)
#     np.save(f'./output/weight_tomo{i}{suffix}.npy', zero_bin)
#     np.save(f'./output/true_tomo{i}{suffix}.npy', zero_bin)

