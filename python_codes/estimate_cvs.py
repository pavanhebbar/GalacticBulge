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
    return 0


def get_netcounthist_fraction(netcount_bins_arr, cand_netcounts_arr,
                              int_netcounts_arr, plot=True,
                              det_names_arr=None):
    """Get histogram of candidate sources based on a property."""
    interested_srcs_histogram = []
    candidate_srcs_histogram = []
    ratios = []
    err_ratios = []
    for i, netcount_bins in enumerate(netcount_bins_arr):
        int_hist = np.histogram(int_netcounts_arr[i], bins=netcount_bins)[0]
        cand_hist = np.histogram(cand_netcounts_arr[i], bins=netcount_bins)[0]
        ratio = cand_hist/int_hist
        err_ratio = (cand_hist**-0.5 + int_hist**-0.5)*ratio
        
        interested_srcs_histogram.append(int_hist)
        candidate_srcs_histogram.append(cand_hist)
        ratios.append(ratio)
        err_ratios.append(err_ratio)
        if not plot:
            return (interested_srcs_histogram, candidate_srcs_histogram,
                    ratios, err_ratios)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Net counts')
    ax1.set_xscale('log')
    ax1.set_xlim(250,1.0E+5)
    ax1.set_ylabel('# per net count bin')
    ax1.hist(int_netcounts_arr, bins=netcount_bins)
    ax1.set_legend(det_names_arr)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Fraction')
    ax2.set_ylim(0, 1.0)
    for i, netcount_bins in enumerate(netcount_bins_arr):
        ax2.errorbar(0.5*(netcount_bins[8:]+netcount_bins[7:-1]),
                     ratios[i][7:],
                     yerr=err_ratios[i][7:], capsize=15)
    ax2.set_legend(det_names_arr)
    return (interested_srcs_histogram, candidate_srcs_histogram, ratios,
            err_ratios)


def get_const(yvals, err):
    nan_yvals = np.isnan(yvals)
    nan_errs = np.isnan(err)
    nan_vals = np.logical_or(nan_yvals, nan_errs)
    return np.sum((yvals/err**2)[~nan_vals])/np.sum((1.0/err**2)[~nan_vals])


def get_cvprob(p_nl_cvsim, p_nl_obs, p1_msp, p1_obs, p1_cv):
    p_nl_cvsim[np.isnan(p_nl_cvsim)] = 0.0
    p1_obs[np.isnan(p1_obs)] = 1.0
    p1_cv[np.isnan(p1_cv)] = 0.0
    return p_nl_cvsim/p_nl_obs*((p1_msp - p1_obs)/(p1_msp - p1_cv))
