"""Python program to analyze and classify XMM spectra."""


import glob2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from astropy.io import fits
from sklearn.decomposition import PCA


class PCASpec(PCA):
    """PCA decomposition of the spectra.

    Highest variance seems to come from the net count, when spectra is
    not normalized
    """
    def __init__(self, n_components, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None):
        super().__init__(
            n_components, copy=copy, whiten=whiten, svd_solver=svd_solver,
            tol=tol, iterated_power=iterated_power, random_state=random_state)

    def spec_transform(self, src_spec, scaled_bgspec=None):
        """Transform spectra to lower dim space."""
        if scaled_bgspec is None:
            scaled_bgspec = np.zeros_like(src_spec)
        return self.transform(src_spec - scaled_bgspec)

    def spec_inv_transform(self, reduced_spec, scaled_bgspec=None):
        """Inverse transform the reduced spec."""
        if scaled_bgspec is None:
            return self.inverse_transform(reduced_spec)
        else:
            return self.inverse_transform(reduced_spec) + scaled_bgspec

    def call(self, src_spec, scaled_bgspec=None):
        """Decompose the spectra using PCA and then inverse transform."""
        if scaled_bgspec is None:
            scaled_bgspec = np.zeros_like(src_spec)
        return (
            self.inverse_transform(self.transform(src_spec - scaled_bgspec)) +
            scaled_bgspec)


def get_enbins_centres(resp_file):
    """Get Energy bins and centres."""
    response = fits.open(resp_file)
    energy_bins = response[2].data
    emin = energy_bins['E_MIN']
    emax = energy_bins['E_MAX']
    e_centres = 0.5*(emin + emax)
    return emin, emax, e_centres


def load_det_spectra(foldername, det, background=False):
    """Load XMM spectra corresponding to given detector."""
    if det == 'PN':
        n_channels = 4096
    else:
        n_channels = 2400
    if foldername[-1] != '/':
        foldername = foldername + '/'
    spec_files = glob2.glob(foldername + '*_' + det + '_combined_src_grp1*.ds')
    spec_arr = np.zeros((len(spec_files), n_channels), dtype=np.float64)
    srcnum_arr = np.zeros(len(spec_files), dtype=int)
    if background:
        bgspec_arr = np.zeros((len(spec_files), n_channels), dtype=np.float64)
    else:
        bgspec_arr = None

    for i, spec_file in enumerate(spec_files):
        spec = fits.open(spec_file)
        spec_arr[i] = spec[1].data['counts']
        srcnum_arr[i] = int(spec_file.split('/')[-1].split('_')[0])
        if background:
            bg_file = (foldername + str(srcnum_arr[i]) + '_' + det +
                       '_combined_bkg_grp.ds')
            bg_spec = fits.open(bg_file)
            bgspec_arr[i] = bg_spec[1].data['counts']*(
                spec[1].header['backscal'] / bg_spec[1].header['backscal'])

    return srcnum_arr, spec_arr, bgspec_arr


def load_xmmspectra_raw(foldername, background=False):
    """Load XMM spectra with background."""
    pn_src_num, pn_specs, pn_bgs = load_det_spectra(foldername, 'PN',
                                                    background)
    mos_src_num, mos_specs, mos_bgs = load_det_spectra(foldername, 'MOS',
                                                       background)
    return ([pn_src_num, mos_src_num], [pn_specs, mos_specs],
            [pn_bgs, mos_bgs])


def bin_spectra(spectra, e_channels, cutoff_en, opt_bin_width):
    """Bin the input spectra with the given bin_width.

    Note that this doesn't ensure min counts in each bin, so that all the
    input spectra have the same energy bins for the classification.
    Input:
    spectra - Input spectra. Can be 1D or 2D array.
    e_channel_min - Min energy bounds of the channels
    e_channel_max - Max energy bounds of the channels
    cutoff_en - Cut off energies for the output spectra. Shouble be an list
        with 2 values
    opt_bin_width - Optimum bin width. The function tries to make the actual
        size to be as close to opt_bin_width as possible. Exact case isn't
        always possible given that the spectra is also binned.
    """
    print(cutoff_en)
    # Perform sanity checks
    if cutoff_en[0] > cutoff_en[1]:
        raise ValueError("Min cut off energy greater than max cutoff energy")
    if cutoff_en[0] > e_channels[-1]:
        raise ValueError('Min cut off energy greater than max energy')
    if cutoff_en[1] < e_channels[0]:
        raise ValueError('Max cut off energy smaller than min energy')
    if ((isinstance(opt_bin_width, list) or
         isinstance(opt_bin_width, np.ndarray)) and (
             np.sum(opt_bin_width) < (cutoff_en[1] - cutoff_en[0]))):
        raise ValueError('Sum of bin widths must be greater than the' +
                         'energy interval needed.')

    cutoff_min_arg = np.searchsorted(e_channels, cutoff_en[0])
    cutoff_max_arg = np.searchsorted(e_channels, cutoff_en[1])
    if e_channels[cutoff_max_arg] > cutoff_en[1]:
        cutoff_max_arg -= 1

    # Merging e_channel_min and e_channel_max into single array of energy bins
    e_channel_bins = e_channels[cutoff_min_arg:cutoff_max_arg+1]

    # Get energy bins according to input bin_widths
    if isinstance(opt_bin_width, (float, np.float)):
        num_bins = int((e_channel_bins[-1] - e_channel_bins[0])/opt_bin_width)
        opt_bin_en = (e_channel_bins[0] +
                      np.arange(0, num_bins+1)*opt_bin_width)
    elif isinstance(opt_bin_width, (list, np.ndarray)):
        opt_bin_en = e_channel_bins[0] + np.cumsum(opt_bin_width)
        opt_bin_en = np.append(e_channel_bins[0], opt_bin_en.copy())
        opt_bin_en = opt_bin_en[np.where(opt_bin_en <= e_channel_bins[-1])]

    # Adjust opt_bin to match with the boundaries of e_channel_bins
    opt_bin_args = np.searchsorted(e_channel_bins, opt_bin_en)
    for i, args in enumerate(opt_bin_args):
        if opt_bin_en[i] != e_channel_bins[args]:
            diff_lower = opt_bin_en[i] - e_channel_bins[args-1]
            diff_up = e_channel_bins[args] - e_channel_bins[i]
            if diff_lower < 0 or diff_up < 0:
                raise ValueError('Something is wrong. Check the differences')
            if diff_lower < diff_up:
                opt_bin_args[i] = args - 1

    if len(spectra.shape) == 1:
        binned_spectra = np.zeros(len(opt_bin_args) - 1, dtype=np.float64)
    elif len(spectra.shape) == 2:
        binned_spectra = np.zeros((len(spectra), len(opt_bin_args) - 1),
                                  dtype=np.float64)
    else:
        raise ValueError("Shape of spectra can only be 1D or 2D")
    for i, args in enumerate(opt_bin_args[:-1]):
        if len(spectra.shape) == 1:
            binned_spectra[i] = np.sum(
                spectra[cutoff_min_arg+args:cutoff_min_arg+opt_bin_args[i+1]])
        else:
            binned_spectra[:, i] = np.sum(
                spectra[:,
                        cutoff_min_arg+args:cutoff_min_arg+opt_bin_args[i+1]],
                axis=1)

    return (e_channel_bins[opt_bin_args], binned_spectra, opt_bin_args[0],
            opt_bin_args[-1])


def norm_spectra(src_spec, bg_spec):
    """Calculate net spectra and normalize the spectra"""
    net_spec = src_spec - bg_spec
    if len(src_spec.shape) == 1:
        net_counts = np.sum(net_counts)
        norm_spec = net_spec/net_counts
    else:
        net_counts = np.sum(net_spec, axis=1)
        norm_spec = (net_spec.transpose()/net_counts).transpose()
    return net_spec, net_counts, norm_spec


def refine_spectra(src_num, src_netspec, src_netcounts, min_counts):
    """Set a minimum count cutoff."""
    refined_mask = src_netcounts >= min_counts
    src_netspec_refined = src_netspec[refined_mask]
    src_netcounts_refined = src_netcounts[refined_mask]
    src_normspec_refined = (src_netspec_refined.transpose() /
                            src_netcounts_refined).transpose()
    return src_netspec_refined, src_netcounts_refined, src_normspec_refined


def spec_loss(src_spec, model_spec, bg_spec=None, weights=None):
    """Poisson related loss between model spectra and source spectra.
    
    If inputs are normalized spectra, use the net_counts/energy_bin as weights.
    """
    sr_spec = src_spec.copy()
    if bg_spec is None:
        bg_spec = np.zeros_like(model_spec)  # Bg is zero if no background
    if weights is None:
        weights = np.ones(len(model_spec), dtype=float)  # Equal weights
    recon_src_spec = model_spec + bg_spec
    if np.any(recon_src_spec <= 0.0):
        print("Some values of model spec are negative.")
        print("Converting them to 1.0E-16.")
        recon_src_spec[recon_src_spec <= 0.0] = 1.0E-16
    if np.any(src_spec == 0):
        print("Zero values in source spec are taken as 1.0E-16")
        sr_spec[src_spec <= 0.0] = 1.0E-16
    loss_arr = recon_src_spec - sr_spec - sr_spec*np.log(recon_src_spec /
                                                         sr_spec) # Loss arr
    loss_arr_weighted = np.multiply(
        loss_arr, weights.reshape(len(src_spec), 1)) # Weighted loss
    loss = 2*np.sum(loss_arr_weighted)/len(src_spec) # C-statistic
    return loss


def compare_pca(src_spec, bg_spec=None, num_comps=None, weights=None):
    """Compare PCA decomposition for different number of nodes."""
    if bg_spec is None:
        bg_spec = np.zeros_like(src_spec)
    if num_comps is None:
        num_comps = [2, 3, 4, 7, 10, 14, 21, 32, 47, 70, 103, 151]      # Number of components
    pca_list = []
    pca_transformed_specs = []
    loss_arr = np.zeros(len(num_comps), dtype=np.float64)
    aicc_arr = np.zeros_like(loss_arr)
    pca_recon_spec = np.zeros((len(num_comps), len(src_spec),
                               len(src_spec[0])), dtype=np.float64)
    for i, ncomp in enumerate(num_comps):
        pca_xmm = PCASpec(n_components=ncomp)
        pca_xmm.fit(src_spec - bg_spec)
        pca_recon_spec[i] = pca_xmm.call(src_spec, bg_spec)
        pca_transformed_specs.append(pca_xmm.spec_transform(src_spec, bg_spec))
        loss = spec_loss(src_spec, pca_recon_spec[i], weights=weights)
        loss_arr[i] = loss
        aicc_arr[i] = loss + 2*ncomp + 2*(ncomp**2 + ncomp)/(189 - ncomp - 1)
        pca_list.append(pca_xmm)
    return (num_comps, pca_list, loss_arr, aicc_arr, pca_transformed_specs,
            pca_recon_spec)