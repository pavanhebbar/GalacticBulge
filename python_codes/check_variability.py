"""Python program to check for short term and long term variability."""

import numpy as np
import pandas as pd
from astropy.io import fits
import scipy.stats as stats


def test_mean_consistency_weighted(mu_values, sigma_values):
    """Chi-square test for consistency of means, using 1/var weighting."""
    weights = 1 / sigma_values**2
    mu_weighted = np.sum(weights * mu_values) / np.sum(weights)
    chi2_mu = np.sum((mu_values - mu_weighted)**2 * weights)
    p_value = 1 - stats.chi2.cdf(chi2_mu, df=len(mu_values) - 1)
    return mu_weighted, chi2_mu, p_value


def process_xmm_variability(fits_file):
    """Return columns to check for short and long term variability."""
    xmm_var_data = fits.getdata(fits_file)
    xmm_srcid = xmm_var_data['SRCID']
    xmm_detid = xmm_var_data['DETID']
    xmm_flux = xmm_var_data['EP_4_FLUX'] + xmm_var_data['EP_5_FLUX']
    xmm_flux_err = (xmm_var_data['EP_4_FLUX_ERROR'] +
                    xmm_var_data['EP_5_FLUX_ERROR'])
    xmm_src_flux = xmm_var_data['SC_EP_4_FLUX'] + xmm_var_data['SC_EP_5_FLUX']
    xmm_src_flux_err = (xmm_var_data['SC_EP_4_FLUX_ERROR'] +
                        xmm_var_data['SC_EP_5_FLUX_ERROR'])
    xmm_ep_chi2prob = xmm_var_data['EP_CHI2PROB']
    # xmm_var_flag == xmm_var_data['VAR_FLAG']
    return (xmm_srcid, xmm_detid, xmm_flux, xmm_flux_err, xmm_src_flux,
            xmm_src_flux_err, xmm_ep_chi2prob)


def process_chandra_variability(txt_file):
    """Return the columns to check for variability of Chandra sources."""
    chandra_var_data = pd.read_csv(txt_file, delimiter='\t', header=25)
    ch_userid = chandra_var_data['usrid']
    ch_name = chandra_var_data['name']
    ch_flux = chandra_var_data['flux_aper_h'].astype(float)
    ch_flux_err_lohi = np.column_stack(
        [ch_flux - chandra_var_data['flux_aper_lolim_h'].astype(float),
         chandra_var_data['flux_aper_hilim_h'].astype(float) - ch_flux])
    ch_obsids = chandra_var_data['obsid']
    ch_obi = chandra_var_data['obi']
    ch_regionid = chandra_var_data['region_id']
    ch_theta = chandra_var_data['theta'].astype(float)
    ch_obs_flux = chandra_var_data['flux_aper_h.1'].astype(float)
    ch_obs_flux_err_lohi = np.column_stack(
        [ch_obs_flux - chandra_var_data['flux_aper_lolim_h.1'].astype(float),
         chandra_var_data['flux_aper_hilim_h.1'].astype(float) - ch_obs_flux])
    ch_var_prob_obs = chandra_var_data['var_prob_h'].astype(float)
    ch_ks_prob_obs = chandra_var_data['ks_prob_h'].astype(float)
    ch_kp_prob_obs = chandra_var_data['kp_prob_h'].astype(float)
    return ([ch_userid, ch_name, ch_obsids, ch_obi, ch_regionid, ch_theta],
            [ch_flux, ch_obs_flux, ch_flux_err_lohi, ch_obs_flux_err_lohi],
            [ch_var_prob_obs, ch_ks_prob_obs, ch_kp_prob_obs])


def check_long_var(srcids, flux_vals, flux_err_vals,
                   src_flux_vals=None, src_flux_err_vals=None):
    """Check for variability between observations."""
    srcids_unique = np.unique(srcids)
    src_const_pval = np.zeros(len(srcids_unique), dtype=float)
    src_chi2_vals = np.zeros(len(srcids_unique), dtype=float)
    src_num_obs = np.zeros(len(srcids_unique), dtype=int)
    for i, src in enumerate(srcids_unique):
        src_args = np.where(srcids == src)[0]
        src_fluxes = flux_vals[src_args]
        src_flux_errs = flux_err_vals[src_args]
        mean_flux, src_chi2_vals[i], src_const_pval[i] = (
            test_mean_consistency_weighted(src_fluxes, src_flux_errs))
        src_num_obs[i] = len(src_args)
        if src_flux_vals is not None and src_flux_err_vals is not None:
            min_src_flux = src_flux_vals[i] - 3*src_flux_err_vals[i]
            max_src_flux = src_flux_vals[i] + 3*src_flux_err_vals[i]
            if mean_flux < min_src_flux or mean_flux > max_src_flux:
                print("Something wrong")
                print("Mean flux: " + str(mean_flux))
                print(r'Src Flux: ' + str(src_flux_vals[i]) + r'$\pm$' +
                      str(src_flux_err_vals[i]))
    src_long_var_flag = (src_const_pval < 0.001)
    return (srcids_unique, src_long_var_flag, src_chi2_vals, src_num_obs,
            src_const_pval)


def check_short_var(srcids, ep_chi2prob, var_flags=None):
    """Check for short term variability."""
    srcids_unique = np.unique(srcids)
    min_ep_prob = np.ones(len(srcids_unique), dtype=float)
    for i, src in enumerate(srcids_unique):
        if var_flags is None:
            src_args = np.where(srcids == src)[0]
        else:
            src_args = np.where(np.logical_and(srcids == src,
                                               var_flags != 'N'))[0]
        min_ep_prob[i] = np.min(ep_chi2prob[src_args])
    src_short_var_flag = (min_ep_prob < 0.000333)
    return (srcids_unique, src_short_var_flag, min_ep_prob)


def chandra_short_var(ch_names, ch_var_obs_p, ch_ks_var_p, ch_kp_var_p):
    """Check for short term variability of the sources within observation.

    Atleast two shoud be greater than 0.999.
    """
    names_unique = np.unique(ch_names)
    src_short_var_flag = np.zeros(len(names_unique), dtype=bool)
    for i, src in enumerate(names_unique):
        obs_args = np.where(ch_names == src)[0]
        for arg in obs_args:
            probs = np.array([ch_var_obs_p[arg], ch_ks_var_p[arg],
                              ch_kp_var_p[arg]])
            num_97 = len(np.where(probs >= 0.999)[0])
            if num_97 >= 2:
                src_short_var_flag[i] = True
                break
    return names_unique, src_short_var_flag


def chandra_long_var(ch_names, flux_vals, flux_err_vals_lohi):
    """Check for long term variability based on chi2 test of const. flux."""
    names_unique = np.unique(ch_names)
    flux_err_vals = np.max(flux_err_vals_lohi, axis=1)
    const_pval = np.zeros(len(names_unique), dtype=float)
    chi2_vals = np.zeros(len(names_unique), dtype=float)
    num_obs = np.zeros(len(names_unique), dtype=int)
    for i, src in enumerate(names_unique):
        obs_args = np.where(ch_names == src)[0]
        flux_vals_src = flux_vals[obs_args]
        flux_errs_src = flux_err_vals[obs_args]
        chi2_vals[i], const_pval[i] = test_mean_consistency_weighted(
            flux_vals_src, flux_errs_src)[1:]
        num_obs[i] = len(obs_args)
    long_var_flag = const_pval < 0.001
    return (names_unique, const_pval, chi2_vals, num_obs, long_var_flag)
