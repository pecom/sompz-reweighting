import numpy as np
from scipy.integrate import trapezoid


spline_bins = np.linspace(0, 5, 101)


obs_counts = np.load('./output/counts.npy')
true_counts = np.load('./output/true_counts.npy')

obs_pdf = np.load('./output/tomo1.npy')
true_pdf = np.load('./output/true_tomo1.npy')

# obs_nz = obs_pdf/obs_counts[0]
# true_nz = true_pdf/true_counts[0] * 20

obs_nz = obs_pdf/trapezoid(obs_pdf, spline_bins[:-1])
true_nz = true_pdf/trapezoid(true_pdf, spline_bins[:-1])

print((obs_nz - true_nz)[:20])

print("Obs", trapezoid(obs_nz, spline_bins[:-1]))
print("True", trapezoid(true_nz, spline_bins[:-1]))
