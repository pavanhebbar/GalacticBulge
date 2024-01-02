"""Program to analyze the colors of bright MSPs in XMM Galactic bulge."""


import os
import glob2
import xspec
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.optimize import fsolve


# Plot functions
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


def plotline_scatter(xdatas, ydatas, pl_types=None, axs=None, xlabel=None,
                     ylabel=None, pl_labels=None, styles=None,
                     yscale='linear', title=None, ylim=None):
    """Plot line and scatter plots."""
    if pl_types is None:
        pl_types = ['line']*len(ydatas)
    if axs is None:
        fig, axs = plt.subplots()
    if pl_labels is None:
        pl_labels = [None]*len(ydatas)
    if styles is None:
        styles = ['-']*len(ydatas)
    axs.set_title(title)
    print(ylim)
    if ylim is not None:
        axs.set_ylim(ylim[0], ylim[1])
    axs.set_xlabel(xlabel)
    axs.set_xscale('log')
    axs.set_yscale(yscale)
    axs.set_ylabel(ylabel)
    for i, pl_type in enumerate(pl_types):
        if pl_type == 'line':
            axs.plot(xdatas[i], ydatas[i], linestyle=styles[i],
                     label=pl_labels[i])
        elif pl_type == 'scatter':
            if styles[i] == '-':
                styles[i] = 'o'
            axs.scatter(xdatas[i], ydatas[i], marker=styles[i],
                        label=pl_labels[i])
        else:
            raise ValueError("pl_type can only be 'line' or 'scatter'")

    axs.legend()
    return axs


def plothist(data_arr, data2=None, axs=None, bins=None, xlabel=None,
             ylabel=None, pl_labels=None, pl_type=None, cbar_label=None):
    """Plot 1D and 2D histograms."""
    if axs is None:
        fig, axs = plt.subplots()
    if pl_type is None:
        pl_type = 'withoutkde'
    if pl_labels is None and isinstance(data_arr[0], (list, np.ndarray)):
        pl_labels = [None]*len(data_arr)
    axs.set_xlabel(xlabel)
    if data2 is None:
        if pl_type == 'withkde':
            if ylabel is None:
                ylabel = 'Density per bin'
            axs.set_ylabel(ylabel)
            axs.hist(data_arr, bins=bins, density=True, label=pl_labels)[:-1]
            if not isinstance(data_arr[0], (list, np.ndarray)):
                data_arr = [data_arr]
            for i, data in enumerate(data_arr):
                if pl_labels[i] is not None:
                    sns.kdeplot(x=data, label=pl_labels[i]+'_kde',
                                ax=axs)
                else:
                    sns.kdeplot(x=data, ax=axs)

            axs.legend()
        else:
            if ylabel is None:
                ylabel = '# per bin'
            axs.set_ylabel(ylabel)
            axs.hist(data_arr, bins=bins,
                     label=pl_labels)[:-1]
            axs.legend()
    else:
        # 2D hist
        axs.set_ylabel(ylabel)
        if bins is None:
            bins = 10
        if pl_type == 'withkde':
            histplot = axs.hist2d(data_arr, data2, bins=bins, density=True)[-1]
            sns.kdeplot(x=data_arr, y=data2, ax=axs)
            cbar = plt.colorbar(mappable=histplot, ax=axs)
            if cbar_label is None:
                cbar_label = 'Density per bin'
            cbar.set_label(cbar_label)
        else:
            histplot = axs.hist2d(data_arr, data2)[-1]
            cbar = plt.colorbar(mappable=histplot, ax=axs)
            if cbar_label is None:
                cbar_label = '# per bin'
            cbar.set_label(cbar_label)
    return axs


def plot_subplots(numrows, numcols, xdatas_arr, ydatas_arr, title=None,
                  subp_types_arr=None, pl_types_arr=None, xlabel_arr=None,
                  ylabel_arr=None, pl_labels_arr=None, styles_arr=None,
                  bins_arr=None, cbar_labels=None, yscale_arr=None,
                  ylim_arr=None):
    """Plot subplots"""
    print(ylim_arr)
    if subp_types_arr is None:
        subp_types_arr = [['linescatter']*numcols]*numrows
    if pl_labels_arr is None:
        pl_labels_arr = [[None]*numcols]*numrows
    if xlabel_arr is None:
        xlabel_arr = [[None]*numcols]*numrows
    if ylabel_arr is None:
        ylabel_arr = [[None]*numcols]*numrows
    if styles_arr is None:
        styles_arr = [[None]*numcols]*numrows
    if bins_arr is None:
        bins_arr = [[None]*numcols]*numrows
    if cbar_labels is None:
        cbar_labels = [[None]*numcols]*numrows
    if yscale_arr is None:
        yscale_arr = [['linear']*numcols]*numrows
    if ylim_arr is None:
        ylim_arr = [[None]*numcols]*numrows

    # Checking if all the arrays are 2D
    if not isinstance(subp_types_arr[0], list):
        subp_types_arr = list(map(list, zip(*[subp_types_arr])))
    if not isinstance(pl_labels_arr[0], list):
        pl_labels_arr = list(map(list, zip(*[pl_labels_arr])))
    if not isinstance(xlabel_arr[0], list):
        xlabel_arr = list(map(list, zip(*[xlabel_arr])))
    if not isinstance(ylabel_arr[0], list):
        ylabel_arr = list(map(list, zip(*[ylabel_arr])))

    # Calling the figure and axes
    fig, axes = plt.subplots(numrows, numcols)
    fig.suptitle(title)
    axes.reshape(numrows, numcols)

    for i in range(numrows):
        for j in range(numcols):
            if subp_types_arr[i][j] == 'linescatter':
                plotline_scatter(
                    xdatas_arr[i][j], ydatas_arr[i][j], pl_types_arr[i][j],
                    axs=axes[i][j], xlabel=xlabel_arr[i][j],
                    ylabel=ylabel_arr[i][j], pl_labels=pl_labels_arr[i][j],
                    styles=styles_arr[i][j], yscale=yscale_arr[i][j],
                    ylim=ylim_arr[i][j])
            elif subp_types_arr[i][j] == 'hist':
                plothist(xdatas_arr[i][j], ydatas_arr[i][j], axs=axes[i][j],
                         bins=bins_arr[i][j], xlabel=xlabel_arr[i][j],
                         ylabel=ylabel_arr[i][j],
                         pl_labels=pl_labels_arr[i][j],
                         pl_type=pl_types_arr[i][j],
                         cbar_label=cbar_labels[i][j])
            else:
                print('subplot type can be linescatter or hist')
    plt.tight_layout()
    return fig, axes


def filter_specs(net_counts, bg_counts, min_netcounts=None, maxbg_ratio=None,
                 det_mask=None):
    """Filter specs."""
    bgratio = bg_counts/net_counts
    filter_mask = det_mask.copy()
    if det_mask is None:
        det_mask = np.ones(len(net_counts), dtype=bool)
    if min_netcounts is not None:
        filter_mask[net_counts < min_netcounts] = False
    if maxbg_ratio is not None:
        filter_mask[bgratio > maxbg_ratio] = False
    return filter_mask


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
    source_nums = np.zeros(num_sources, dtype=object)
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
            source_nums[i] = (
                mos_specfiles[i-len(pn_bgspecs)].split('/')[-1].split('_')[0])
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
            if counts_enrange < floor_counts:
                counts_enrange = floor_counts
        else:
            counts_enrange[counts_enrange < floor_counts] = floor_counts
    return counts_enrange, elow_index, ehigh_index


def get_counts_enrange2(spec, en_range, ebin_channels, floor_counts=None):
    """Another method to get counts"""
    en_mask = np.logical_and(ebin_channels >= en_range[0],
                             ebin_channels <= en_range[1])
    elow_index = np.where(en_mask)[0][0]
    ehigh_index = np.where(en_mask)[0][-1]
    if len(spec.shape) == 1:
        counts_enrange = np.sum(spec[en_mask])
    else:
        counts_enrange = np.sum(spec[:, en_mask], axis=1)
    if floor_counts is not None:
        if len(spec.shape) == 1:
            if counts_enrange < floor_counts:
                floor_counts = 0
        else:
            counts_enrange[counts_enrange < floor_counts] = 0
    return counts_enrange, elow_index, ehigh_index


def plot_spec_summary(sim_src_spec, sim_bg_spec, obs_src_spec, obs_bg_spec,
                      ebins, det='', en_range=None, plot=True,
                      det_mask_sim_orig=None, det_mask_obs_orig=None):
    """Plot spec summary."""
    if en_range is None:
        en_range = [0.2, 10.0]
    net_spec_sim = sim_src_spec - sim_bg_spec
    net_counts_sim, en_lowindex, en_highindex = get_counts_enrange(
        net_spec_sim, en_range, ebins)
    if det_mask_sim_orig is None:
        det_mask_sim = np.ones(len(sim_src_spec), dtype=bool)
    else:
        det_mask_sim = det_mask_sim_orig.copy()
    det_mask_sim[net_counts_sim < 1] = False
    net_spec_sim = net_spec_sim[:, en_lowindex:en_highindex]
    bg_counts_sim = get_counts_enrange(sim_bg_spec, en_range, ebins)[0]
    norm_spec_sim = (
        net_spec_sim.transpose()/net_counts_sim).transpose()[det_mask_sim]

    net_spec_obs = obs_src_spec - obs_bg_spec
    net_counts_obs = get_counts_enrange(net_spec_obs, en_range, ebins)[0]
    if det_mask_obs_orig is None:
        det_mask_obs = np.ones(len(sim_src_spec), dtype=bool)
    else:
        det_mask_obs = det_mask_obs_orig.copy()
    det_mask_obs[net_counts_obs < 1] = False
    net_spec_obs = net_spec_obs[:, en_lowindex:en_highindex]
    bg_counts_obs = get_counts_enrange(obs_bg_spec, en_range, ebins)[0]
    # src_counts_obs = get_counts_enrange(obs_src_spec, en_range, ebins)[0]
    norm_spec_obs = (
        net_spec_obs.transpose()/net_counts_obs).transpose()[det_mask_obs]
    if plot is False:
        return ([net_counts_sim, net_counts_obs],
                [bg_counts_sim, bg_counts_obs], en_lowindex, en_highindex)

    plotline_scatter([ebins[en_lowindex:en_highindex],
                      ebins[en_lowindex:en_highindex]],
                     [median_filter(np.mean(norm_spec_sim, axis=0), size=10),
                      median_filter(np.mean(norm_spec_obs, axis=0), size=10)],
                     xlabel='Energy (keV)',
                     ylabel='Mean smoothed normalized spectra',
                     pl_labels=['Simuated MSPs', 'Observed sources'],
                     title='Simulated and Observed ' + det + ' spectra')
    plot_subplots(
        2, 2, [[[np.log10(net_counts_sim[det_mask_sim]),
                 np.log10(net_counts_obs[det_mask_obs])],
                [np.log10(bg_counts_sim/net_counts_sim)[det_mask_sim],
                 np.log10(bg_counts_obs/net_counts_obs)[det_mask_obs]]],
               [np.log10(net_counts_sim[det_mask_sim]),
                np.log10(net_counts_obs[det_mask_obs])]],
        [[None, None], [np.log10(bg_counts_sim[det_mask_sim]),
                        np.log10(bg_counts_obs[det_mask_obs])]],
        title='Net, Bg counts in observed, simulated ' + det + ' spectra',
        subp_types_arr=[['hist', 'hist'], ['hist', 'hist']],
        pl_types_arr=[['withkde', 'withkde'], ['withkde', 'withkde']],
        xlabel_arr=[['log(Net counts)', 'log(Bg counts/Net counts)'],
                    ['log(Simulated net counts)', 'log(Observed net counts)']],
        ylabel_arr=[['# per bin', '# per bin'], ['log (Bg counts)',
                                                 'log(Bg counts)']],
        pl_labels_arr=[[['Simulated MSPs', 'Observed sources'],
                        ['Simulated MSPs', 'Observed sources']], [None, None]])
    return ([net_counts_sim, net_counts_obs],
            [bg_counts_sim, bg_counts_obs], en_lowindex, en_highindex)


def get_enbins_centres(resp_file):
    """Get Energy bins and centres."""
    response = fits.open(resp_file)
    energy_bins = response[2].data
    emin = energy_bins['E_MIN']
    emax = energy_bins['E_MAX']
    e_centres = 0.5*(emin + emax)
    return emin, emax, e_centres


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
    median_colors = np.zeros(len(srcprop_bins)-1)
    colors_std = np.zeros(len(srcprop_bins)-1)
    for i, bin_edge in enumerate(srcprop_bins[:-1]):
        mean_colors[i], colors_std[i], median_colors[i] = get_colors_perbin(
            colors, src_prop, [bin_edge, srcprop_bins[i+1]])
    return mean_colors, colors_std, median_colors


def get_colors_perbin(colors, src_prop, bin_edges):
    """Get expected color for each bin"""
    mask = np.where(np.logical_and(np.logical_and(src_prop >= bin_edges[0],
                                                  src_prop < bin_edges[1]),
                                   np.isfinite(colors)))
    # print(len(mask[0]))
    bin_colors = colors[mask]
    return np.mean(bin_colors), np.std(bin_colors), np.median(bin_colors)


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


def bin_colors_withprop(colors, prop_list, prop_listnames, obs_colors,
                        obs_props, prop_binslist=None, plot=True, plot_det=''):
    """Bin colors with properties."""
    mean_colors_withprop = []
    median_colors_withprop = []
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
        mean_colors, colors_std, median_colors = get_colors_binned(
            colors, prop_list[i], prop_binslist[i])
        mean_colors_withprop.append(mean_colors)
        std_colors_withprop.append(colors_std)
        median_colors_withprop.append(median_colors)
        if plot:
            plot_subplots(
                2, 2, [[[prop_list[i]], [0.5*(prop_binslist[i][1:] +
                                              prop_binslist[i][:-1]),
                                         0.5*(prop_binslist[i][1:] +
                                              prop_binslist[i][:-1])]],
                       [[prop_list[i], 0.5*(prop_binslist[i][1:] +
                                            prop_binslist[i][:-1]),
                         0.5*(prop_binslist[i][1:] + prop_binslist[i][:-1])],
                        [obs_props[i], 0.5*(prop_binslist[i][1:] +
                                            prop_binslist[i][:-1]),
                         0.5*(prop_binslist[i][1:] + prop_binslist[i][:-1])]]],
                [[[colors], [mean_colors, median_colors]],
                 [[colors, mean_colors+colors_std, mean_colors+3*colors_std],
                  [obs_colors, mean_colors+colors_std, mean_colors+colors_std]]
                 ],
                subp_types_arr=[['linescatter', 'linescatter'],
                                ['linescatter', 'linescatter']],
                pl_types_arr=[[['scatter'], ['line', 'line']],
                              [['scatter', 'line', 'line'],
                               ['scatter', 'line', 'line']]],
                xlabel_arr=[[prop_listnames[i], prop_listnames[i]+' bins'],
                            [prop_listnames[i], prop_listnames[i]]],
                ylabel_arr=[['Colors', 'Mean/median color'],
                            ['Simulated sources', 'Observed colors']],
                pl_labels_arr=[[None, ['Mean color', 'Median color']],
                               [None, ['Colors', 'Mean + 1.0std',
                                       'Mean + 3.0std']]],
                title='Color vs ' + prop_listnames[i] + ' for ' + plot_det,
                ylim_arr=[[(0, 5), (0, 5)], [(0, 5), (0, 5)]])
            '''
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
            '''
    return (prop_binslist, mean_colors_withprop, std_colors_withprop,
            median_colors_withprop)


def process_singledet(det_ebins, det_srcspecs, det_bgspecs, det_mask, det_name,
                      det_obs_ebins=None, det_obs_srcspecs=None,
                      det_obs_bgspecs=None, det_obsmask=None, range_fe=None):
    """Process single detector data."""
    ([netcounts, obs_netcounts],
     [bgcounts, obs_bgcounts], lowindex, highindex) = plot_spec_summary(
        det_srcspecs, det_bgspecs, det_obs_srcspecs, det_obs_bgspecs,
        det_ebins, en_range=[2.0, 10.0], det_mask_sim=det_mask,
        det_mask_obs=det_obsmask)
    ([netcounts_aroundfe, obs_netcounts_aroundfe],
     [bgcounts_aroundfe, obs_bgcounts_aroundfe], aroundfe_lowindex,
     aroundfe_highindex) = plot_spec_summary(
        det_srcspecs, det_bgspecs, det_obs_srcspecs, det_obs_bgspecs,
        det_ebins, en_range=[5.8, 7.6], det_mask_sim=det_mask,
        det_mask_obs=det_obsmask, det=det_name)
    # netcounts, bgcounts, lowindex, highindex = plot_spec_summary(
    #    det_srcspecs, det_bgspecs, det_ebins, det=det_name,
    #    en_range=[2.0, 10.0], det_mask_sim=det_mask)
    # if sim_data:
    #    print('hi')
    #    filter_mask = filter_specs(netcounts_aroundfe, bgcounts_aroundfe,
    #                               det_mask=det_mask, maxbg_ratio=10)
    #    print(np.where(filter_mask != det_mask))
    #    det_mask = filter_mask
    # plot_spec_summary(
    #    det_srcspecs, det_bgspecs, det_ebins, det=det_name,
    #    en_range=[5.8, 7.6], det_mask=det_mask)
    bg_net_ratio = bgcounts/netcounts
    obs_bg_netratio = obs_bgcounts/obs_netcounts
    bg_net_ratio_aroundfe = bgcounts_aroundfe/netcounts_aroundfe
    obs_bg_netratio_aroundfe = obs_bgcounts_aroundfe/obs_netcounts_aroundfe
    ([fe_net, fe_src, fe_bg], [cont1_net, cont1_src, cont1_bg],
     [cont2_net, cont2_src, cont2_bg]) = get_line_cont_counts(
        det_srcspecs, det_bgspecs, det_ebins, range_fe=range_fe)
    ([obs_fe_net, obs_fe_src, obs_fe_bg],
     [obs_cont1_net, obs_cont1_src, obs_cont1_bg],
     [obs_cont2_net, obs_cont2_src, obs_cont2_bg]) = get_line_cont_counts(
        det_obs_srcspecs, det_obs_bgspecs, det_obs_ebins, range_fe=range_fe)
    det_colors = get_colors_basic(fe_net, cont1_net, cont2_net, mask=det_mask)
    obs_det_colors = get_colors_basic(obs_fe_net, obs_cont1_net, obs_cont2_net)
    netcount_bins = 10**(np.linspace(1, 5, 21))
    netcount_aroundfe_bins = 10**(np.linspace(0, 4, 21))
    bg_net_ratio_bins = 10**(np.linspace(-2, 1, 21))
    bg_net_ratio_aroundfe_bins = 10**(np.linspace(-2, 1, 21))
    (mean_colors_withprop, std_colors_withprop,
     median_colors_withprop) = bin_colors_withprop(
        det_colors, [netcounts, netcounts_aroundfe, bg_net_ratio,
                     bg_net_ratio_aroundfe],
        ['Net counts (2-10 keV)', 'Net counts (5.8-7.6 keV)',
         'Bg/net ratio (2-10 keV)', 'Bg/net ratio (5.8-7.6 keV)'],
        obs_det_colors, [obs_netcounts, obs_netcounts_aroundfe,
                         obs_bg_netratio, obs_bg_netratio_aroundfe],
        [netcount_bins, netcount_aroundfe_bins, bg_net_ratio_bins,
         bg_net_ratio_aroundfe_bins], plot=True, plot_det=det_name)[1:]
    sim_det_dict = {'counts_2_10': [netcounts, bgcounts],
                    'counts_aroundfe': [netcounts_aroundfe, bgcounts],
                    'lowhigh_indices': [lowindex, highindex,
                                        aroundfe_lowindex, aroundfe_highindex],
                    'fe_cont_netcounts': [fe_net, cont1_net, cont2_net],
                    'fe_cont_bgcounts': [fe_bg, cont1_bg, cont2_bg],
                    'fe_cont_srccounts': [fe_src, cont1_src, cont2_src],
                    'prop_bins': [netcount_bins, netcount_aroundfe_bins,
                                  bg_net_ratio_bins,
                                  bg_net_ratio_aroundfe_bins],
                    'mean_std_colors_withprop': [
                        mean_colors_withprop, std_colors_withprop,
                        median_colors_withprop]}
    obs_det_dict = {'counts_2_10': [obs_netcounts, obs_bgcounts],
                    'counts_aroundfe': [obs_netcounts_aroundfe,
                                        obs_bgcounts_aroundfe],
                    'lowhigh_indices': [lowindex, highindex,
                                        aroundfe_lowindex, aroundfe_highindex],
                    'fe_cont_netcounts': [obs_fe_net, obs_cont1_net,
                                          obs_cont2_net],
                    'fe_cont_bgcounts': [obs_fe_bg, obs_cont1_bg,
                                         obs_cont2_bg],
                    'fe_cont_srccounts': [obs_fe_src, obs_cont1_src,
                                          obs_cont2_src]}
    return sim_det_dict, obs_det_dict


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


def combine_wantedprops(pn_dict, mos_dict, pn_mask, mos_mask):
    """Combine the counts from PN and MOS detectors."""
    net_counts = combine_pn_mos_prop(
        pn_dict['counts_2_10'][0], mos_dict['counts_2_10'][0], pn_mask,
        mos_mask)
    bg_counts = combine_pn_mos_prop(
        pn_dict['counts_2_10'][1], mos_dict['counts_2_10'][1], pn_mask,
        mos_mask)
    net_counts_aroundfe = combine_pn_mos_prop(
        pn_dict['counts_aroundfe'][0], mos_dict['counts_aroundfe'][0], pn_mask,
        mos_mask)
    bg_counts_aroundfe = combine_pn_mos_prop(
        pn_dict['counts_aroundfe'][1], mos_dict['counts_aroundfe'][1], pn_mask,
        mos_mask)
    combined_fe_net = combine_pn_mos_prop(
        pn_dict['fe_cont_netcounts'][0], mos_dict['fe_cont_netcounts'][0],
        pn_mask, mos_mask)
    combined_cont1_net = combine_pn_mos_prop(
        pn_dict['fe_cont_netcounts'][1], mos_dict['fe_cont_netcounts'][1],
        pn_mask, mos_mask)
    combined_cont2_net = combine_pn_mos_prop(
        pn_dict['fe_cont_netcounts'][2], mos_dict['fe_cont_netcounts'][2],
        pn_mask, mos_mask)
    combined_colors = get_colors_basic(combined_fe_net, combined_cont1_net,
                                       combined_cont2_net)
    return (net_counts, bg_counts/net_counts, net_counts_aroundfe,
            bg_counts_aroundfe/net_counts_aroundfe, combined_colors)


def process_given_sim_nhtype(nhtype, range_fe=None):
    """Process a given type of NH"""
    pn_emin, pn_emax, pn_ecentres = get_enbins_centres(
        '../XMM_responses/PN/epn_bu23_dY9.rmf.gz')
    pn_ebins = np.append(pn_emin, pn_emax[-1])
    mos_emin, mos_emax, mos_ecentres = get_enbins_centres(
        '../XMM_responses/MOS_5eV/m1_e10_im_p0_c.rmf')
    mos_ebins = np.append(mos_emin, mos_emax[-1])
    print('Loaded Energy bins')
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
    print('Loaded Simulated and observed data')
    sim_pn_dict, obs_pn_dict = process_singledet(
        pn_ebins, sim_pn_specs, sim_pn_bgs, sim_pn_mask, det_name='PN',
        det_obs_ebins=pn_ebins, det_obs_srcspecs=obs_pn_specs,
        det_obs_bgspecs=obs_pn_bgs, det_obsmask=obs_pn_mask, range_fe=range_fe)
    sim_mos_dict, obs_mos_dict = process_singledet(
        mos_ebins, sim_mos_specs, sim_mos_bgs, sim_mos_mask, det_name='MOS',
        det_obs_ebins=mos_ebins, det_obs_srcspecs=obs_mos_specs,
        det_obs_bgspecs=obs_mos_bgs, det_obsmask=obs_mos_mask,
        range_fe=range_fe)
    print('Processed each detectors individually')
    (sim_netcounts, sim_bgratio, sim_netcounts_aroundfe, sim_bgratio_aroundfe,
     sim_combined_colors) = combine_wantedprops(sim_pn_dict, sim_mos_dict,
                                                sim_pn_mask, sim_mos_mask)
    (obs_netcounts, obs_bgratio, obs_netcounts_aroundfe, obs_bgratio_aroundfe,
     obs_combined_colors) = combine_wantedprops(obs_pn_dict, obs_mos_dict,
                                                obs_pn_mask, obs_mos_mask)
    netcount_bins = 10**(np.linspace(1, 5, 21))
    netcount_aroundfe_bins = 10**(np.linspace(0, 4, 21))
    bg_net_ratio_bins = 10**(np.linspace(-2, 1, 21))
    bg_net_ratio_aroundfe_bins = 10**(np.linspace(-2, 1, 21))
    (mean_colors_withprop, std_colors_withprop,
     median_colors_withprop) = bin_colors_withprop(
        sim_combined_colors, [sim_netcounts, sim_bgratio,
                              sim_netcounts_aroundfe, sim_bgratio_aroundfe],
        ['Combined net counts', 'Combined BG/Net count ratio',
         'Combined net counts around Fe',
         'Combined Bg/Net count ratio around Fe'],
        obs_combined_colors, [obs_netcounts, obs_bgratio,
                              obs_netcounts_aroundfe, obs_bgratio_aroundfe],
        prop_binslist=[netcount_bins, bg_net_ratio_bins,
                       netcount_aroundfe_bins, bg_net_ratio_aroundfe_bins],
        plot=True, plot_det='PN+MOS')[1:]
    combined_dict = {'individual_dicts': [sim_pn_dict, sim_mos_dict,
                                          obs_pn_dict, obs_mos_dict],
                     'spectra': [sim_pn_specs, sim_mos_specs, obs_pn_specs,
                                 obs_mos_specs],
                     'bg_spectra': [sim_pn_bgs, sim_mos_bgs, obs_pn_bgs,
                                    obs_mos_bgs],
                     'combined_counts': [sim_netcounts, sim_netcounts_aroundfe,
                                         obs_netcounts,
                                         obs_netcounts_aroundfe],
                     'combined_bgratios': [sim_bgratio, sim_bgratio_aroundfe,
                                           obs_bgratio, obs_bgratio_aroundfe],
                     'colors': [sim_combined_colors, obs_combined_colors],
                     'prop_bins': [netcount_bins, netcount_aroundfe_bins,
                                   bg_net_ratio_bins,
                                   bg_net_ratio_aroundfe_bins],
                     'mean_std_colors_withprop': [mean_colors_withprop,
                                                  std_colors_withprop,
                                                  median_colors_withprop]}
    return source_nums, combined_dict


def get_data(ntype):

