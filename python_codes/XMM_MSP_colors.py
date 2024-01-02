"""Program to analyze the colors of bright MSPs in XMM Galactic bulge."""


import os
import glob2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.optimize import fsolve


def set_plotparams(plottype):
    """Set plot parameters."""
    if plottype == 'notebook':
        plt.rcParams["figure.figsize"] = (20, 15)
        plt.rcParams["axes.titlesize"] = 24
        plt.rcParams["axes.labelsize"] = 24
        plt.rcParams["lines.linewidth"] = 3
        plt.rcParams["lines.markersize"] = 10
        plt.rcParams["xtick.labelsize"] = 20
        plt.rcParams["ytick.labelsize"] = 20
        plt.rcParams["legend.fontsize"] = 20
        plt.rcParams['xtick.major.size'] = 16
        plt.rcParams['xtick.minor.size'] = 8
        plt.rcParams['ytick.major.size'] = 16
        plt.rcParams['ytick.minor.size'] = 8
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['xtick.direction'] = 'inout'
        plt.rcParams['ytick.direction'] = 'inout'
    elif plottype == 'presentation':
        plt.rcParams["figure.figsize"] = (4, 3)
        plt.rcParams["axes.titlesize"] = 16
        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["lines.linewidth"] = 3
        plt.rcParams["lines.markersize"] = 5
        plt.rcParams["xtick.labelsize"] = 12
        plt.rcParams["ytick.labelsize"] = 12
        plt.rcParams["legend.fontsize"] = 12
        plt.rcParams['xtick.major.size'] = 8
        plt.rcParams['xtick.minor.size'] = 4
        plt.rcParams['ytick.major.size'] = 8
        plt.rcParams['ytick.minor.size'] = 4
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['xtick.direction'] = 'inout'
        plt.rcParams['ytick.direction'] = 'inout'
    elif plottype == 'paper':
        plt.rcParams["figure.figsize"] = (5, 5)
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["lines.linewidth"] = 2
        plt.rcParams["lines.markersize"] = 2
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["legend.fontsize"] = 10
        plt.rcParams['xtick.major.size'] = 6
        plt.rcParams['xtick.minor.size'] = 3
        plt.rcParams['ytick.major.size'] = 6
        plt.rcParams['ytick.minor.size'] = 3
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'


def load_sim_xmmspec(folder, background=False, numsim=20000, basename=None):
    """Load simulated XMM MOS and PN spectra."""
    if basename is None:
        basename = ''
    pn_specs = np.zeros((numsim, 4096), dtype=np.float64)
    pn_bgspecs = np.zeros((numsim, 4096), dtype=np.float64)
    mos_specs = np.zeros((numsim, 2400), dtype=np.float64)
    mos_bgspecs = np.zeros((numsim, 2400), dtype=np.float64)
    pn_mask = np.zeros(numsim, dtype=bool)
    mos_mask = np.zeros(numsim, dtype=bool)
    for i in range(numsim):
        pnfile = folder + basename + str(i) + '_PN.fak'
        mosfile = folder + basename + str(i) + '_MOS.fak'
        if os.path.isfile(pnfile):
            pn_mask[i] = True
            spec_data = fits.open(pnfile)
            pn_specs[i] = spec_data[1].data['counts']
            if background:
                bgfile = pnfile[:-4] + '_bkg.fak'
                bgspec_data = fits.open(bgfile)
                pn_bgspecs[i] = bgspec_data[1].data['counts']*(
                    spec_data[1].header['backscal'] /
                    bgspec_data[1].header['backscal'])
        if os.path.isfile(mosfile):
            mos_mask[i] = True
            spec_data = fits.open(mosfile)
            mos_specs[i] = spec_data[1].data['counts']
            if background:
                bgfile = mosfile[:-4] + '_bkg.fak'
                bgspec_data = fits.open(bgfile)
                mos_bgspecs[i] = bgspec_data[1].data['counts']*(
                    spec_data[1].header['backscal'] /
                    bgspec_data[1].header['backscal'])
    return ([pn_mask, mos_mask], [pn_specs, mos_specs],
            [pn_bgspecs, mos_bgspecs])


def filter_specs(net_counts, bg_counts, min_netcounts=None, maxbg_ratio=None,
                 det_mask=None):
    """Filter specs."""
    bgratio = bg_counts/net_counts
    filter_mask = det_mask.copy()
    if min_netcounts is not None:
        filter_mask[net_counts < min_netcounts] = False
    if maxbg_ratio is not None:
        filter_mask[bgratio > maxbg_ratio] = False
    return filter_mask


def load_xmmspec_observed(folder, background=False):
    """Load Observed XMM spec."""
    pn_specfiles = glob2.glob(folder + '*PN_combined_src_grp.ds')
    mos_specfiles = glob2.glob(folder + '*MOS_combined_src_grp.ds')
    num_combined = len(glob2.glob(folder + '*PN_MOS_combined_src.png'))
    num_sources = len(pn_specfiles) + len(mos_specfiles) - num_combined
    pn_specs = np.zeros((num_sources, 4096), dtype=float)
    pn_mask = np.zeros(num_sources, dtype=bool)
    mos_specs = np.zeros((num_sources, 2400), dtype=float)
    mos_mask = np.zeros(num_sources, dtype=bool)
    pn_bgspecs = np.zeros((num_sources, 4096), dtype=float)
    mos_bgspecs = np.zeros((num_sources, 2400), dtype=float)
    source_nums = np.zeros(num_sources, dtype=np.object)
    for i in range(num_sources):
        if i < len(pn_specfiles):
            pn_mask[i] = True
            source_nums[i] = pn_specfiles[i].split('/')[-1].split('_')[0]
            spec_data = fits.open(pn_specfiles[i])
            pn_specs[i] = spec_data[1].data['counts']
            if background:
                bgfile = pn_specfiles[i][:-10] + 'bkg_grp.ds'
                bg_data = fits.open(bgfile)
                pn_bgspecs[i] = bg_data[1].data['counts']*(
                    spec_data[1].header['backscal'] /
                    bg_data[1].header['backscal'])
            if os.path.isfile(folder + source_nums[i] +
                              '_MOS_combined_src_grp.ds'):
                mosfile = folder + source_nums[i] + '_MOS_combined_src_grp.ds'
                spec_data = fits.open(mosfile)
                mos_mask[i] = True
                mos_specfiles.remove(mosfile)
                mos_specs[i] = spec_data[1].data['counts']
                if background:
                    bgfile = mosfile[:-10] + 'bkg_grp.ds'
                    bg_data = fits.open(bgfile)
                    mos_bgspecs[i] = bg_data[1].data['counts']*(
                        spec_data[1].header['backscal'] /
                        bg_data[1].header['backscal'])
        else:
            mos_mask[i] = True
            spec_data = fits.open(mos_specfiles[i-len(pn_bgspecs)])
            mos_specs[i] = spec_data[1].data['counts']
            if background:
                bgfile = mosfile[:-10] + 'bkg_grp.ds'
                bg_data = fits.open(bgfile)
                mos_bgspecs[i] = bg_data[1].data['counts']*(
                    spec_data[1].header['backscal'] /
                    bg_data[1].header['backscal'])
    return (source_nums, [pn_mask, pn_specs, pn_bgspecs],
            [mos_mask, mos_specs, mos_bgspecs])


def plot_spec_summary(src_spec, bg_spec, ebins, det='', en_range=None,
                      plot=True, det_mask=None):
    """Plot spectra summary"""
    if en_range is None:
        en_range = [0.2, 10.0]
    net_spec = src_spec - bg_spec
    net_counts, en_lowindex, en_highindex = get_counts_enrange(
        net_spec, en_range, ebins)
    if det_mask is None:
        det_mask = np.ones(len(src_spec), dtype=bool)
    det_mask[net_counts < 1] = False
    net_spec = net_spec[:, en_lowindex:en_highindex]
    bg_counts = get_counts_enrange(bg_spec, en_range, ebins)[0]
    src_counts = get_counts_enrange(src_spec, en_range, ebins)[0]
    norm_spec = (net_spec.transpose()/net_counts).transpose()[det_mask]
    if plot is False:
        return net_counts, bg_counts, en_lowindex, en_highindex

    plt.figure()
    plt.title('Normalized mean spectra of ' + det + ' data')
    plt.xlabel(det + ' Energy (keV)')
    plt.ylabel('Normalized mean spectra')
    plt.plot(
        0.5*(ebins[en_lowindex:en_highindex] +
             ebins[en_lowindex+1:en_highindex+1]),
        median_filter(np.mean(norm_spec, axis=0), size=10))
    plt.xscale('log')

    plt.figure()
    plt.title('Histogram of net counts and background counts ' + det + ' data')
    plt.xlabel(r'log$_{10}$ (' + det + ' counts)')
    plt.ylabel('# per bin')
    plt.hist([np.log10(net_counts)[det_mask],
              np.log10(bg_counts)[det_mask]],
             bins=20)
    plt.legend(['Net counts', 'Scaled Background counts'])
    plt.figure()
    plt.xlabel(r'log$_{10}$ (' + det + ' background counts/' + det +
               ' net counts)')
    plt.ylabel('# per bin')
    plt.hist(np.log10(bg_counts/net_counts)[det_mask], bins=20)
    plt.figure()
    plt.xlabel(r'log$_{10}$ (' + det + ' background counts/' + det +
               ' source+bg counts)')
    plt.ylabel('# per bin')
    plt.hist(np.log10(bg_counts/src_counts)[det_mask], bins=20)
    plt.figure()
    plt.hist2d(np.log10(net_counts[det_mask]),
               np.log10(bg_counts/net_counts)[det_mask], bins=20)
    plt.xlabel(r'log$_{10}$(' + det + ' net counts)')
    plt.ylabel(r'log$_{10}$(' + det + ' bg counts/' + det + ' net counts)')
    plt.colorbar()
    plt.figure()
    plt.xlabel(r'log$_{10}$(' + det + ' net counts)')
    plt.ylabel(r'log$_{10}$(' + det + ' bg counts/' + det + ' net counts)')
    sns.kdeplot(x=np.log10(net_counts[det_mask]),
                y=np.log10((bg_counts/net_counts)[det_mask]),
                fill=True)

    return net_counts, bg_counts, en_lowindex, en_highindex


def get_enbins_centres(resp_file):
    """Get Energy bins and centres."""
    response = fits.open(resp_file)
    energy_bins = response[2].data
    emin = energy_bins['E_MIN']
    emax = energy_bins['E_MAX']
    e_centres = 0.5*(emin + emax)
    return emin, emax, e_centres


def get_counts_enrange(spec, en_range, ebin_channels, floor_counts=None):
    """Get counts in the given energy range."""
    elow_index = np.where(ebin_channels >= en_range[0])[0][0]
    ehigh_index = np.where(ebin_channels <= en_range[1])[0][-1]
    if len(spec.shape) == 1:
        counts_enrange = np.sum(spec[elow_index:ehigh_index])
    else:
        counts_enrange = np.sum(spec[:, elow_index:ehigh_index], axis=1)
    if floor_counts is not None:
        if len(spec.shape) == 1:
            counts_enrange = floor_counts
        else:
            counts_enrange[counts_enrange < 0] = floor_counts
    return counts_enrange, elow_index, ehigh_index


def get_line_cont_counts(src_spec, bg_spec, ebins, net_spec=None,
                         range_fe=None, cont1_range=None, cont2_range=None):
    """Get line and continuum counts"""
    if net_spec is None:
        net_spec = src_spec - bg_spec
    if range_fe is None:
        range_fe = [6.2, 7.2]
    if cont1_range is None:
        cont1_range = [5.8, 6.2]
    if cont2_range is None:
        cont2_range = [7.2, 7.6]
    sim_fe_net = get_counts_enrange(net_spec, range_fe, ebins,
                                    floor_counts=0)[0]
    sim_fe_src = get_counts_enrange(src_spec, range_fe, ebins,
                                    floor_counts=0)[0]
    sim_fe_bg = get_counts_enrange(bg_spec, range_fe, ebins,
                                   floor_counts=0)[0]

    sim_cont1_net = get_counts_enrange(net_spec, cont1_range, ebins,
                                       floor_counts=0)[0]
    sim_cont1_src = get_counts_enrange(src_spec, cont1_range, ebins,
                                       floor_counts=0)[0]
    sim_cont1_bg = get_counts_enrange(bg_spec, cont1_range, ebins,
                                      floor_counts=0)[0]

    sim_cont2_net = get_counts_enrange(net_spec, cont2_range, ebins,
                                       floor_counts=0)[0]
    sim_cont2_src = get_counts_enrange(src_spec, cont2_range, ebins,
                                       floor_counts=0)[0]
    sim_cont2_bg = get_counts_enrange(bg_spec, cont2_range, ebins,
                                      floor_counts=0)[0]
    return ([sim_fe_net, sim_fe_src, sim_fe_bg],
            [sim_cont1_net, sim_cont1_src, sim_cont1_bg],
            [sim_cont2_net, sim_cont2_src, sim_cont2_bg])


def get_colors_basic(counts_line_net, counts_cont1_net, counts_cont2_net,
                     mask=None):
    """Get colors."""
    if mask is None:
        mask = np.ones(len(counts_line_net), dtype=bool)
    colors = counts_line_net/(counts_cont1_net + counts_cont2_net)
    colors[~mask] = 0
    return colors


def get_colors_binned(colors, src_prop, srcprop_bins, mask=None):
    """Get expected color for given bins."""
    mean_colors = np.zeros(len(srcprop_bins)-1)
    colors_std = np.zeros(len(srcprop_bins)-1)
    for i, bin_edge in enumerate(srcprop_bins[:-1]):
        mean_colors[i], colors_std[i] = get_colors_perbin(
            colors, src_prop, [bin_edge, srcprop_bins[i+1]])
    return mean_colors, colors_std


def get_colors_perbin(colors, src_prop, bin_edges):
    """Get expected color for each bin"""
    mask = np.where(np.logical_and(np.logical_and(src_prop >= bin_edges[0],
                                                  src_prop < bin_edges[1]),
                                   np.isfinite(colors)))
    # print(len(mask[0]))
    bin_colors = colors[mask]
    return np.mean(bin_colors), np.std(bin_colors)


def estimate_counts_enrange(kvals, gamma_vals, en_low, en_high):
    """Estimate counts in the energy range for a power law."""
    int_power = gamma_vals - 1
    return kvals/int_power*(1/en_low**int_power - 1/en_high**int_power)


def get_pl_param_resid(x_val, en1_range_counts, en2_range_counts,
                       en1_range=None, en2_range=None):
    """Get residual for given power law parameters."""
    if len(x_val.shape) == 2:
        k_vals, gamma_vals = x_val
    else:
        k_vals = x_val[:int(len(x_val)/2)]
        gamma_vals = x_val[int(len(x_val)/2):]
    if en1_range is None:
        en1_range = [5.8, 6.2]
    if en2_range is None:
        en2_range = [7.2, 7.6]
    expected_en1counts = estimate_counts_enrange(
        k_vals, gamma_vals, en1_range[0], en1_range[1])
    expected_en2counts = estimate_counts_enrange(
        k_vals, gamma_vals, en2_range[0], en2_range[1])
    en1_residual = en1_range_counts - expected_en1counts
    en2_residual = en2_range_counts - expected_en2counts
    if len(x_val.shape) == 2:
        return np.vstack([en1_residual, en2_residual])

    return np.append(en1_residual, en2_residual)


def solve_for_pl(en1_range_counts, en2_range_counts, en1_range=None,
                 en2_range=None):
    """Solve for power-law parameters"""
    if en1_range is None:
        en1_range = [5.8, 6.2]
    if en2_range is None:
        en2_range = [7.2, 7.6]
    k_0 = np.ones(len(en1_range_counts))*en2_range[1]*en2_range_counts*max(
        en2_range[1] - en2_range[0], en1_range[1] - en1_range[0])
    gamma_0 = np.ones(len(en1_range_counts))*1.5
    x_soln = fsolve(get_pl_param_resid, np.append(k_0, gamma_0), args=(
        en1_range_counts, en2_range_counts, en1_range, en2_range))
    k_soln = x_soln[:int(len(x_soln/2))]
    gamma_soln = x_soln[int(len(x_soln/2)):]
    return k_soln, gamma_soln


def estimate_cont_counts_pl(cont1_counts, cont2_counts, line_range,
                            cont1_range=None, cont2_range=None):
    """Estimating the continuum counts range in the line range"""
    k_soln, gamma_soln = solve_for_pl(cont1_counts, cont2_counts, cont1_range,
                                      cont2_range)
    return estimate_counts_enrange(k_soln, gamma_soln, line_range[0],
                                   line_range[1])


def combine_pn_mos_prop(pn_prop, mos_prop, pn_mask, mos_mask):
    """Combine PN and MOS properties."""
    combined_prop = np.zeros_like(pn_prop)
    combined_prop[np.logical_and(
        pn_mask, ~mos_mask)] = pn_prop[np.logical_and(pn_mask, ~mos_mask)]
    combined_prop[np.logical_and(
        ~pn_mask, mos_mask)] = mos_prop[np.logical_and(~pn_mask, mos_mask)]
    combined_prop[np.logical_and(
        pn_mask, mos_mask)] = (pn_prop + mos_prop)[
            np.logical_and(pn_mask, mos_mask)]
    return combined_prop


def bin_colors_withprop(colors, prop_list, prop_listnames,
                        prop_binslist=None, plot=True, plot_det=''):
    """Bin colors with properties."""
    mean_colors_withprop = []
    std_colors_withprop = []
    if prop_binslist is None:
        prop_binslist = []
        no_given_bins = True
    else:
        no_given_bins = False
    for i in range(len(prop_list)):
        print('Calaculating mean and standard deviation in colors vs. ' +
              prop_listnames[i])
        if no_given_bins:
            prop_bins = 10**np.linspace(
                np.log10(max(0.00001, np.min(prop_list[i]))),
                np.log10(min(1.0E+5, np.max(prop_list[i]))), 21)
            prop_binslist.append(prop_bins)
        mean_colors, colors_std = get_colors_binned(colors, prop_list[i],
                                                    prop_binslist[i])
        mean_colors_withprop.append(mean_colors)
        std_colors_withprop.append(colors_std)
        if plot:
            plt.figure()
            plt.title('Scatter plot of color vs ' + prop_listnames[i] + ' for '
                      + plot_det + ' data')
            plt.xlabel(prop_listnames[i] + ' bins')
            plt.xscale('log')
            plt.xlim(np.min(prop_binslist[i]), np.max(prop_binslist[i]))
            plt.ylabel('Colors')
            plt.ylim(0, 10)
            plt.scatter(prop_list[i], colors)
            plt.figure()
            plt.title('Mean color vs ' + prop_listnames[i] + ' for ' + plot_det
                      + ' data')
            plt.xlabel(prop_listnames[i] + ' bins')
            plt.xscale('log')
            plt.ylabel('Mean color')
            plt.plot(0.5*(prop_binslist[i][1:] + prop_binslist[i][:-1]),
                     mean_colors)
            plt.figure()
            plt.title('Standard deviation in color vs ' + prop_listnames[i] +
                      ' for ' + plot_det + ' data')
            plt.xlabel(prop_listnames[i] + ' bins')
            plt.xscale('log')
            plt.ylabel('Standard deviation in color')
            plt.yscale('log')
            plt.plot(0.5*(prop_binslist[i][1:] + prop_binslist[i][:-1]),
                     colors_std)
    return prop_binslist, mean_colors_withprop, std_colors_withprop


def compare_color_real_observed(obs_colors, prop_list, prop_listnames,
                                prop_binslist, mean_color_withprop,
                                std_color_withprop, plot_det):
    """Compare the observed colors with the simulated mean and std."""
    for i, prop in enumerate(prop_list):
        plt.figure()
        plt.title('Scatter plot of observed color vs ' + prop_listnames[i] +
                  ' for ' + plot_det + ' data')
        plt.xlabel(prop_listnames[i] + ' bins')
        plt.xscale('log')
        plt.xlim(np.min(prop_binslist[i]), np.max(prop_binslist[i]))
        plt.ylabel('Colors')
        plt.ylim(0, 5)
        plt.scatter(prop_list[i], obs_colors, label='Observed colors')
        plt.plot(0.5*(prop_binslist[i][1:] + prop_binslist[i][:-1]),
                 mean_color_withprop[i], linestyle='-',
                 label='Mean simulated color')
        plt.plot(0.5*(prop_binslist[i][1:] + prop_binslist[i][:-1]),
                 mean_color_withprop[i] + std_color_withprop[i],
                 linestyle='--', label='Mean + 1sigma')
        plt.plot(0.5*(prop_binslist[i][1:] + prop_binslist[i][:-1]),
                 mean_color_withprop[i] + 3*std_color_withprop[i],
                 linestyle='--', label='Mean + 3sigma')
        plt.legend()


def processing_singledet(det_ebins, det_srcspecs, det_bgspecs, det_mask,
                         det_name, sim_data=True, mean_colors=None,
                         std_colors=None):
    """Processing data for a single detector."""
    netcounts, bgcounts, lowindex, highindex = plot_spec_summary(
        det_srcspecs, det_bgspecs, det_ebins, det=det_name,
        en_range=[2.0, 10.0], det_mask=det_mask)
    (netcounts_aroundfe, bgcounts_aroundfe, aroundfe_lowindex,
     aroundfe_highindex) = plot_spec_summary(
        det_srcspecs, det_bgspecs, det_ebins, det=det_name,
        en_range=[5.8, 7.6], plot=False, det_mask=det_mask)
    if sim_data:
        print('hi')
        filter_mask = filter_specs(netcounts_aroundfe, bgcounts_aroundfe,
                                   det_mask=det_mask, maxbg_ratio=10)
        print(np.where(filter_mask != det_mask))
        det_mask = filter_mask
    plot_spec_summary(
        det_srcspecs, det_bgspecs, det_ebins, det=det_name,
        en_range=[5.8, 7.6], det_mask=det_mask)
    bg_net_ratio = bgcounts/netcounts
    bg_net_ratio_aroundfe = bgcounts_aroundfe/netcounts_aroundfe
    ([fe_net, fe_src, fe_bg], [cont1_net, cont1_src, cont1_bg],
     [cont2_net, cont2_src, cont2_bg]) = get_line_cont_counts(
        det_srcspecs, det_bgspecs, det_ebins)
    det_colors = get_colors_basic(fe_net, cont1_net, cont2_net, mask=det_mask)
    netcount_bins = 10**(np.linspace(1, 5, 21))
    netcount_aroundfe_bins = 10**(np.linspace(0, 4, 21))
    bg_net_ratio_bins = 10**(np.linspace(-2, 1, 21))
    bg_net_ratio_aroundfe_bins = 10**(np.linspace(-2, 1, 21))
    if sim_data:
        mean_colors_withprop, std_colors_withprop = bin_colors_withprop(
            det_colors, [netcounts, netcounts_aroundfe, bg_net_ratio,
                         bg_net_ratio_aroundfe], [
                'Net counts (2-10 keV)', 'Net counts (5.8-7.6 keV)',
                'Bg-to-net ratio (2-10 keV)', 'Bg-to-net ratio (5.8-7.6 keV)'],
            [netcount_bins, netcount_aroundfe_bins, bg_net_ratio_bins,
             bg_net_ratio_aroundfe_bins], plot=True, plot_det=det_name)[1:]
        det_dict = {'counts_2_10': [netcounts, bgcounts],
                    'counts_aroundfe': [netcounts_aroundfe, bgcounts],
                    'lowhigh_indices': [lowindex, highindex,
                                        aroundfe_lowindex, aroundfe_highindex],
                    'fe_cont_netcounts': [fe_net, cont1_net, cont2_net],
                    'fe_cont_bgcounts': [fe_bg, cont1_bg, cont2_bg],
                    'fe_cont_srccounts': [fe_src, cont1_src, cont2_src],
                    'prop_bins': [netcount_bins, netcount_aroundfe_bins,
                                  bg_net_ratio_bins,
                                  bg_net_ratio_aroundfe_bins],
                    'mean_std_colors_withprop': [mean_colors_withprop,
                                                 std_colors_withprop]}
    else:
        compare_color_real_observed(
            det_colors, [netcounts, netcounts_aroundfe, bg_net_ratio,
                         bg_net_ratio_aroundfe], [
                'Net counts (2-10 keV)', 'Net counts (5.8-7.6 keV)',
                'Bg-to-net ratio (2-10 keV)', 'Bg-to-net ratio (5.8-7.6 keV)'],
            [netcount_bins, netcount_aroundfe_bins, bg_net_ratio_bins,
             bg_net_ratio_aroundfe_bins], mean_colors, std_colors, det_name)
        det_dict = {'counts_2_10': [netcounts, bgcounts],
                    'counts_aroundfe': [netcounts_aroundfe, bgcounts],
                    'lowhigh_indices': [lowindex, highindex,
                                        aroundfe_lowindex, aroundfe_highindex],
                    'fe_cont_netcounts': [fe_net, cont1_net, cont2_net],
                    'fe_cont_bgcounts': [fe_bg, cont1_bg, cont2_bg],
                    'fe_cont_srccounts': [fe_src, cont1_src, cont2_src],
                    'prop_bins': [netcount_bins, netcount_aroundfe_bins,
                                  bg_net_ratio_bins,
                                  bg_net_ratio_aroundfe_bins],
                    'mean_std_colors_withprop': [mean_colors, std_colors]}
    return det_dict


def process_given_sim_nhtype(nhtype):
    """Main function to process a given type of NH sources."""
    pn_emin, pn_emax, pn_ecentres = get_enbins_centres(
        '../XMM_responses/PN/epn_bu23_dY9.rmf.gz')
    pn_ebins = np.append(pn_emin, pn_emax[-1])
    mos_emin, mos_emax, mos_ecentres = get_enbins_centres(
        '../XMM_responses/MOS_5eV/m1_e10_im_p0_c.rmf')
    mos_ebins = np.append(mos_emin, mos_emax[-1])
    if nhtype == 'high':
        ([sim_pn_mask, sim_mos_mask], [sim_pn_specs, sim_mos_specs],
         [sim_pn_bgs, sim_mos_bgs]) = load_sim_xmmspec(
             '../data/sim_msps_highNH_PN_MOS/', background=True,
             basename='msp_highNH_')
        (source_nums, [obs_pn_mask, obs_pn_specs, obs_pn_bgs],
         [obs_mos_mask, obs_mos_specs, obs_mos_bgs]) = load_xmmspec_observed(
             '../data/Galactic_highNH_combinedXMM/', background=True)
    elif nhtype == 'mid':
        ([sim_pn_mask, sim_mos_mask], [sim_pn_specs, sim_mos_specs],
         [sim_pn_bgs, sim_mos_bgs]) = load_sim_xmmspec(
             '../data/sim_msps_midNH_PN_MOS/', background=True,
             basename='msp_midNH_')
        (source_nums, [obs_pn_mask, obs_pn_specs, obs_pn_bgs],
         [obs_mos_mask, obs_mos_specs, obs_mos_bgs]) = load_xmmspec_observed(
             '../data/Galactic_midNH_combinedXMM/', background=True)
    elif nhtype == 'low':
        ([sim_pn_mask, sim_mos_mask], [sim_pn_specs, sim_mos_specs],
         [sim_pn_bgs, sim_mos_bgs]) = load_sim_xmmspec(
             '../data/sim_msps_lowNH_PN_MOS/', background=True,
             basename='msp_lowNH_')
        (source_nums, [obs_pn_mask, obs_pn_specs, obs_pn_bgs],
         [obs_mos_mask, obs_mos_specs, obs_mos_bgs]) = load_xmmspec_observed(
             '../data/Galactic_midNH_combinedXMM/', background=True)
    else:
        print('No such type of NH')
    sim_pn_dict = processing_singledet(pn_ebins, sim_pn_specs, sim_pn_bgs,
                                       sim_pn_mask, det_name='PN')
    sim_mos_dict = processing_singledet(mos_ebins, sim_mos_specs, sim_mos_bgs,
                                        sim_mos_mask, det_name='MOS')
    # Combining net and bg counts
    sim_netcounts = combine_pn_mos_prop(
        sim_pn_dict['counts_2_10'][0], sim_mos_dict['counts_2_10'][0],
        sim_pn_mask, sim_mos_mask)
    sim_bgcounts = combine_pn_mos_prop(
        sim_pn_dict['counts_2_10'][1], sim_mos_dict['counts_2_10'][1],
        sim_pn_mask, sim_mos_mask)
    sim_netcounts_aroundfe = combine_pn_mos_prop(
        sim_pn_dict['counts_aroundfe'][0], sim_mos_dict['counts_aroundfe'][0],
        sim_pn_mask, sim_mos_mask)
    sim_bgcounts_aroundfe = combine_pn_mos_prop(
        sim_pn_dict['counts_aroundfe'][1], sim_mos_dict['counts_aroundfe'][1],
        sim_pn_mask, sim_mos_mask)
    sim_bgratio = sim_bgcounts/sim_netcounts
    sim_bgratio_aroundfe = sim_bgcounts_aroundfe/sim_netcounts_aroundfe

    # Combine the Fe line and continuum counts
    sim_combined_fe_net = combine_pn_mos_prop(
        sim_pn_dict['fe_cont_netcounts'][0],
        sim_mos_dict['fe_cont_netcounts'][0], sim_pn_mask, sim_mos_mask)
    sim_combined_cont1_net = combine_pn_mos_prop(
        sim_pn_dict['fe_cont_netcounts'][1],
        sim_mos_dict['fe_cont_netcounts'][1], sim_pn_mask, sim_mos_mask)
    sim_combined_cont2_net = combine_pn_mos_prop(
        sim_pn_dict['fe_cont_netcounts'][2],
        sim_mos_dict['fe_cont_netcounts'][2], sim_pn_mask, sim_mos_mask)
    sim_combined_colors = get_colors_basic(
        sim_combined_fe_net, sim_combined_cont1_net, sim_combined_cont2_net)
    netcount_bins = 10**(np.linspace(1, 5, 21))
    netcount_aroundfe_bins = 10**(np.linspace(0, 4, 21))
    bg_net_ratio_bins = 10**(np.linspace(-2, 1, 21))
    bg_net_ratio_aroundfe_bins = 10**(np.linspace(-2, 1, 21))
    mean_colors_withprop, std_colors_withprop = bin_colors_withprop(
        sim_combined_colors, [sim_netcounts, sim_netcounts_aroundfe,
                              sim_bgratio, sim_bgratio_aroundfe], [
            'Net counts (2-10 keV)', 'Net counts (5.8-7.6 keV)',
            'Bg-to-net ratio (2-10 keV)', 'Bg-to-net ratio (5.8-7.6 keV)'], [
                netcount_bins, netcount_aroundfe_bins, bg_net_ratio_bins,
                bg_net_ratio_aroundfe_bins], plot=True,
        plot_det='PN + MOS')[1:]

    # Repeat for observed data
    obs_pn_dict = processing_singledet(
        pn_ebins, obs_pn_specs, obs_pn_bgs, obs_pn_mask, det_name='obs PN',
        sim_data=False, mean_colors=sim_pn_dict['mean_std_colors_withprop'][0],
        std_colors=sim_pn_dict['mean_std_colors_withprop'][1])
    obs_mos_dict = processing_singledet(
        mos_ebins, obs_mos_specs, obs_mos_bgs, obs_mos_mask,
        det_name='obs MOS', sim_data=False,
        mean_colors=sim_mos_dict['mean_std_colors_withprop'][0],
        std_colors=sim_mos_dict['mean_std_colors_withprop'][1])
    obs_netcounts = combine_pn_mos_prop(
        obs_pn_dict['counts_2_10'][0], obs_mos_dict['counts_2_10'][0],
        obs_pn_mask, obs_mos_mask)
    obs_bgcounts = combine_pn_mos_prop(
        obs_pn_dict['counts_2_10'][1], obs_mos_dict['counts_2_10'][1],
        obs_pn_mask, obs_mos_mask)
    obs_netcounts_aroundfe = combine_pn_mos_prop(
        obs_pn_dict['counts_aroundfe'][0], obs_mos_dict['counts_aroundfe'][0],
        obs_pn_mask, obs_mos_mask)
    obs_bgcounts_aroundfe = combine_pn_mos_prop(
        obs_pn_dict['counts_aroundfe'][1], obs_mos_dict['counts_aroundfe'][1],
        obs_pn_mask, obs_mos_mask)
    obs_bgratio = obs_bgcounts/obs_netcounts
    obs_bgratio_aroundfe = obs_bgcounts_aroundfe/obs_netcounts_aroundfe
    obs_combined_fe_net = combine_pn_mos_prop(
        obs_pn_dict['fe_cont_netcounts'][0],
        obs_mos_dict['fe_cont_netcounts'][0], obs_pn_mask, obs_mos_mask)
    obs_combined_cont1_net = combine_pn_mos_prop(
        obs_pn_dict['fe_cont_netcounts'][1],
        obs_mos_dict['fe_cont_netcounts'][1], obs_pn_mask, obs_mos_mask)
    obs_combined_cont2_net = combine_pn_mos_prop(
        obs_pn_dict['fe_cont_netcounts'][2],
        obs_mos_dict['fe_cont_netcounts'][2], obs_pn_mask, obs_mos_mask)
    obs_combined_colors = get_colors_basic(
        obs_combined_fe_net, obs_combined_cont1_net, obs_combined_cont2_net)
    compare_color_real_observed(
            obs_combined_colors, [obs_netcounts, obs_netcounts_aroundfe,
                                  obs_bgratio, obs_bgratio_aroundfe], [
                'Net counts (2-10 keV)', 'Net counts (5.8-7.6 keV)',
                'Bg-to-net ratio (2-10 keV)', 'Bg-to-net ratio (5.8-7.6 keV)'],
            [netcount_bins, netcount_aroundfe_bins, bg_net_ratio_bins,
             bg_net_ratio_aroundfe_bins], mean_colors_withprop,
            std_colors_withprop, 'obs. PN + MOS')
    combined_dict = {'individual_dicts': [sim_pn_dict, sim_mos_dict,
                                          obs_pn_dict, obs_mos_dict],
                     'spectra': [sim_pn_specs, sim_mos_specs, obs_pn_specs,
                                 obs_mos_specs],
                     'bg_spectra': [sim_pn_bgs, sim_mos_bgs, obs_pn_bgs,
                                    obs_mos_bgs],
                     'combined_counts': [sim_netcounts, sim_netcounts_aroundfe,
                                         obs_netcounts,
                                         obs_netcounts_aroundfe],
                     'combined_bgcounts': [sim_bgcounts, sim_bgcounts_aroundfe,
                                           sim_bgratio, sim_bgratio_aroundfe,
                                           obs_bgcounts, obs_bgcounts_aroundfe,
                                           obs_bgratio, obs_bgratio_aroundfe],
                     'colors': [sim_combined_colors, obs_combined_colors],
                     'fe_cont_netcounts': [sim_combined_fe_net,
                                           sim_combined_cont1_net,
                                           sim_combined_cont2_net],
                     'prop_bins': [netcount_bins, netcount_aroundfe_bins,
                                   bg_net_ratio_bins,
                                   bg_net_ratio_aroundfe_bins],
                     'mean_std_colors_withprop': [mean_colors_withprop,
                                                  std_colors_withprop]}
    return source_nums, combined_dict
