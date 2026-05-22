import numpy as np
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

suffix = config['suffix']
print(f"Using {suffix=:}")
N_pdf_bins = config['N_pdf_bins']

true = np.load(f'./output/true_pdfs{suffix}.npy')
base = np.load(f'./output/base_pdfs{suffix}.npy')
weight = np.load(f'./output/weight_pdfs{suffix}.npy')

print(true[1,3:6])
print(base[1,3:6])
print(weight[1,3:6])

# np.save(f'./output/base_pdfs{suffix}.npy', zero_bin)
# np.save(f'./output/weight_pdfs{suffix}.npy', zero_bin)
# np.save(f'./output/true_pdfs{suffix}.npy', zero_bin)

# for i in range(1,5):
#     np.save(f'./output/base_tomo{i}{suffix}.npy', zero_bin)
#     np.save(f'./output/weight_tomo{i}{suffix}.npy', zero_bin)
#     np.save(f'./output/true_tomo{i}{suffix}.npy', zero_bin)

