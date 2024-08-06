""""Python program to estimate the number of line sources."""


from pyexpat import model
import numpy as np
import matplotlib.pyplot as plt


def read_fit_stats(file):
    """"Read spectral fitting statistics."""
    fits_stats_table = np.loadtxt(file)
    source_num = fits_stats_table[:, 0]
    cstat_vals = fits_stats_table[:, 1].astype(float)
    dof_vals = fits_stats_table[:, -1].astype(float)
    log_cvm_vals = fits_stats_table[:, 2].astype(float)
    goodness_vals = fits_stats_table[:, 3].astype(float)
    return source_num, cstat_vals, dof_vals, log_cvm_vals, goodness_vals


def model_params_file(file, model_type='pl')
    """"Red parameters of the best fitting model."""
    model_params_table = np.loadtxt(file)
    source_num = model_params_table[:, 0]
    nh_vals = model_params_table[:, 1].astype(float)
    pl_index_vals = model_params_table[:, 2].astype(float)
    pl_norm_vals = model_params_table[:, 3].astype(float)
    if model_type == 'pl':
        return source_num, nh_vals, pl_index_vals, pl_norm_vals
    if model_type == 'pl_gauss':
        line_pos_vals = model_params_table[:, 4]
        line_pos_low = model_params_table[:, 5]
        line_pos_high = model_params_table[:, 6]
        line_norm_vals = model_params_table[:, 7]
        line_norm_low = model_params_table[:, 8]
        line_norm_high = model_params_table[:, 9]
        return (source_num, nh_vals, pl_index_vals, pl_norm_vals,
                line_pos_vals, line_norm_vals, line_pos_low, line_pos_high,
                line_norm_low, line_norm_high)
    if model_type == 'pl_gauss3':
        line1_norm_vals = model_params_table[:, 4]
        line2_norm_vals = model_params_table[:, 5]
        line3_norm_vals = model_params_table[:, 6]
        line1_norm_low = model_params_table[:, 7]
        line1_norm_high = model_params_table[:, 8]
        line2_norm_low = model_params_table[:, 9]
        line2_norm_high = model_params_table[:, 10]
        line3_norm_low = model_params_table[:, 11]
        line3_norm_high = model_params_table[:, 12]
        return (source_num, nh_vals, pl_index_vals, pl_norm_vals,
                line1_norm_vals, line2_norm_vals, line3_norm_vals,
                line1_norm_low, line2_norm_low, line3_norm_low,
                line1_norm_high, line2_norm_high, line2_norm_high)
