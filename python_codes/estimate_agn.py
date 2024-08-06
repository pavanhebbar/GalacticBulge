"""Python program to estimate the number of background AGN.

Calculate in galactic coordinates and convert to equatorial coordinates.
"""

import copy
import os
import glob2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from matplotlib.colors import LogNorm
from scipy import interpolate


def gal_to_eq(gal_l, gal_b):
    """Convert from galactic to equatorial coordinates.

    gal_l and gal_b in degrees.
    """
    coord_gal = SkyCoord(gal_l*u.degree, gal_b*u.degree, frame='galactic')
    ra_icrs = coord_gal.icrs.ra.value
    dec_icrs = coord_gal.icrs.dec.value
    return ra_icrs, dec_icrs


def eq_to_gal(ra_frame, dec_frame, frame='icrs'):
    """Convert from celestial to galactic coordinates."""
    coord_eq = SkyCoord(ra=ra_frame*u.degree, dec=dec_frame*u.degree,
                        frame=frame)
    gal_l = coord_eq.galactic.l.value
    gal_b = coord_eq.galactic.b.value
    return gal_l, gal_b


def read_nhdata(filename, gal_l_column='Long', gal_b_column='Lat',
                nh_column='N_H', num_grids=80):
    """Read absorption column data."""
    nh_data = pd.read_csv(filename)
    gal_l_vals = np.array(nh_data[gal_l_column])
    gal_b_vals = np.array(nh_data[gal_b_column])
    nh_vals = np.array(nh_data[nh_column])
    gal_l_2d = gal_l_vals.reshape(num_grids, num_grids)  # Transpose of normal
    # i.e l values are increasing along the row and constant in all columns
    gal_b_2d = gal_b_vals.reshape(num_grids, num_grids)  # Transpose of normal
    gal_l_1d_grid = np.hstack(
        [gal_l_2d[0, 0] - 0.5*(gal_l_2d[1, 0] - gal_l_2d[0, 0]),
         0.5*(gal_l_2d[1:, 0] + gal_l_2d[:-1, 0]),
         gal_l_2d[-1, 0] + 0.5*(gal_l_2d[-1, 0] - gal_l_2d[-2, 0])])
    gal_b_1d_grid = np.hstack(
        [gal_b_2d[0, 0] - 0.5*(gal_b_2d[0, 1] - gal_b_2d[0, 0]),
         0.5*(gal_b_2d[0, 1:] + gal_b_2d[0, :-1]),
         gal_b_2d[0, -1] + 0.5*(gal_b_2d[0, -1] - gal_b_2d[0, -2])])
    nh_grid = nh_vals.reshape(num_grids, num_grids)
    # Returned values are in proper convention, l values constant across rows
    # and b values  constant across columns
    return (np.meshgrid(gal_l_1d_grid, gal_b_1d_grid),
            (gal_l_2d.transpose(), gal_b_2d.transpose()), nh_grid.transpose())


def get_grid_indices(gal_l_grid, gal_b_grid, gal_coordinates):
    """Get indices on where the coordinates lie in the grid.

    gal_l_grid and gal_b_grid must have l increasing along the columns and b
    increasing along the rows
    """
    gal_l_1dgrid = gal_l_grid[0, :]
    gal_b_1dgrid = gal_b_grid[:, 0]
    corr_gal_l = gal_coordinates[:, 0].copy()
    corr_gal_l[corr_gal_l > 180] = corr_gal_l[corr_gal_l > 180] - 360
    index_l = np.searchsorted(gal_l_1dgrid, corr_gal_l) - 1
    index_b = np.searchsorted(gal_b_1dgrid, gal_coordinates[:, 1]) - 1
    return index_l, index_b


def get_coordinates_exposure(specfile):
    """Get the coordinates and the exposure of the given spectrum."""
    spec_data = fits.open(specfile)
    src_ra = float(spec_data[1].header['SRC_CRA'])
    src_dec = float(spec_data[1].header['SRC_CDEC'])
    src_exposure = float(spec_data[1].header['EXPOSURE'])
    src_l, src_b = eq_to_gal(src_ra, src_dec)
    return np.array([src_l, src_b]), src_exposure


def get_source_name(specfile):
    """Get name of the source whose spectra is given."""
    return specfile.split('/')[-1].split('_')[0]


def get_name_coord_exp_list(file_list):
    """Get eposures and coordinates of a list of spectra."""
    coord_list = np.zeros((len(file_list), 2), dtype=float)
    exp_list = np.zeros(len(file_list), dtype=float)
    source_name_list = np.zeros(len(file_list), dtype=object)
    for i, file in enumerate(file_list):
        coord_list[i], exp_list[i] = get_coordinates_exposure(file)
        source_name_list[i] = get_source_name(file)
    coord_list[:, 0][coord_list[:, 0] > 180] = (
        coord_list[:, 0][coord_list[:, 0] > 180] - 360.0)
    return source_name_list, coord_list, exp_list


def map_exposures(coord_list, exp_list, gal_l_grid, gal_b_grid):
    """Map exposures to coordinates."""
    l_indices, b_indices = get_grid_indices(gal_l_grid, gal_b_grid, coord_list)
    num_l_grid, num_b_grid = np.array(gal_l_grid.shape) - 1
    exp_map = np.zeros((num_l_grid, num_b_grid), dtype=float)
    for i in range(num_l_grid):
        for j in range(num_b_grid):
            bin_indices = np.where(np.logical_and(
                l_indices == i, b_indices == j))[0]
            if len(bin_indices) > 0:
                exp_map[i, j] = np.max(exp_list[bin_indices])

    return exp_map


def wcs_to_gal_grid(wcs_header):
    """Get a galactic grid corresponding to the wcs file."""
    ax1_len, axis2_len = wcs_header.pixel_shape
    axis1_pixels_1d = np.arange(ax1_len)
    axis2_pixels_1d = np.arange(axis2_len)
    axis1_grid, axis2_grid = np.meshgrid(axis1_pixels_1d, axis2_pixels_1d)
    gal_grid_skycoords = wcs_header.pixel_to_world(axis1_grid,
                                                   axis2_grid).galactic
    gal_l_grid = gal_grid_skycoords.l.value
    gal_b_grid = gal_grid_skycoords.b.value
    return gal_l_grid, gal_b_grid


def read_expmap(expmap_file):
    """Read exposure map."""
    hdu_table = fits.open(expmap_file)
    exp_map = hdu_table[0].data
    exp_filter = hdu_table[0].header['FILTER']
    exp_map_coord = wcs_to_gal_grid(WCS(hdu_table[0].header))
    return exp_map_coord, exp_map, exp_filter


def expmap_on_grid(expmap_file, full_grid_l, full_grid_b):
    """Export exposure map on the grid."""
    exp_map_coord, exp_map, exp_filter = read_expmap(expmap_file)
    gal_l_coordinates = exp_map_coord[0].reshape((-1, 1))
    gal_b_coordinates = exp_map_coord[1].reshape((-1, 1))
    l_indices, b_indices = get_grid_indices(
        full_grid_l, full_grid_b,
        np.hstack([gal_l_coordinates, gal_b_coordinates]))
    grid_count = np.zeros_like(full_grid_l)
    exp_map_grid = np.zeros_like(full_grid_b)
    for i, l_index in enumerate(l_indices):
        grid_count[b_indices[i], l_index] += 1
        exp_map_grid[b_indices[i], l_index] += exp_map.reshape(-1)[i]
    grid_count[np.logical_and(grid_count == 0, exp_map_grid == 0)] = 1
    return (exp_map_grid/grid_count)[:-1, :-1], exp_filter


def combine_exp_maps(exp_map_list, full_grid_l, full_grid_b, nofilter=False):
    """Add exposure maps."""
    if nofilter:
        exp_map_nofilter = np.zeros_like(full_grid_l[:-1, :-1])
    exp_map_thin = np.zeros_like(full_grid_l[:-1, :-1])
    exp_map_medium = np.zeros_like(full_grid_l[:-1, :-1])
    exp_map_thick = np.zeros_like(full_grid_l[:-1, :-1])
    for exp_map in exp_map_list:
        exp_map_grid, exp_filter = expmap_on_grid(exp_map, full_grid_l,
                                                  full_grid_b)
        if exp_filter.lower() == 'thin':
            exp_map_thin += exp_map_grid
        elif exp_filter.lower() == 'medium':
            exp_map_medium += exp_map_grid
        elif exp_filter.lower() == 'thick':
            exp_map_thick += exp_map_grid
        elif nofilter:
            print(exp_map_grid)
            exp_map_nofilter += exp_map_grid
    if nofilter:
        return exp_map_nofilter
    return [exp_map_thin, exp_map_medium, exp_map_thick]


def get_grid_centers(grid_corners_l, grid_corners_b):
    """Get centers of the grid."""
    grid_centers_l = 0.25*(grid_corners_l[1:, 1:] + grid_corners_l[1:, :-1] +
                           grid_corners_l[:-1, 1:] + grid_corners_l[:-1, :-1])
    grid_centers_b = 0.25*(grid_corners_b[1:, 1:] + grid_corners_b[1:, :-1] +
                           grid_corners_b[:-1, 1:] + grid_corners_b[:-1, :-1])
    return grid_centers_l, grid_centers_b


def get_grid_corners_1d(grid_centers_1d):
    """Get corners of the grid from the centers."""
    grid_corners_1d = np.hstack([
        grid_centers_1d[0] - 0.5*(grid_centers_1d[1] - grid_centers_1d[0]),
        0.5*(grid_centers_1d[1:] + grid_centers_1d[:-1]),
        grid_centers_1d[-1] + 0.5*(grid_centers_1d[-1] - grid_centers_1d[-2])])
    return grid_corners_1d


def main_expmaps(folder=None, pattern='*', num_grids=60, full_grid_l=None,
                 full_grid_b=None, plot=False, telescope='XMM'):
    """Main function."""
    if full_grid_l is None or full_grid_b is None:
        grid_1d = np.linspace(-3, 3, num_grids+1)
        full_grid_l, full_grid_b = np.meshgrid(grid_1d, grid_1d.copy())
    exp_map_list = glob2.glob(folder + '/' + pattern)
    print(exp_map_list)
    grid_centers_l, grid_centers_b = get_grid_centers(full_grid_l, full_grid_b)
    if telescope == 'Chandra':
        exp_map = combine_exp_maps(exp_map_list, full_grid_l, full_grid_b,
                                   nofilter=True)
    elif telescope == 'XMM':
        exp_map_thin, exp_map_medium, exp_map_thick = combine_exp_maps(
            exp_map_list, full_grid_l, full_grid_b)
    else:
        raise ValueError('Telescope can only be XMM or Chandra')
    if not plot:
        if telescope == 'Chandra':
            return (exp_map/358.134, [grid_centers_l, grid_centers_b], None,
                    None)
        return ([exp_map_thin, exp_map_medium, exp_map_thick],
                [grid_centers_l, grid_centers_b], None, None)

    exp_levels = np.logspace(3, 6.7, 100)
    if telescope == 'Chandra':
        fig, axes = plt.subplots(1, 2)
        plot_contours(grid_centers_l, grid_centers_b, exp_map/358.134,
                      filled=True, axes=axes[0], map_levels=exp_levels,
                      map_label='Exposure (s)',
                      title='Combined exposure [ACIS]')
        return exp_map/358.134, [grid_centers_l, grid_centers_b], fig, axes

    fig, axes = plt.subplots(2, 2)
    det_filters = ['Thin', 'Thick', 'Medium']
    for i, exp_map in enumerate([exp_map_thin, exp_map_medium,
                                 exp_map_thick]):
        plot_contours(grid_centers_l, grid_centers_b, exp_map, filled=True,
                      axes=axes[np.unravel_index(i, (2, 2))],
                      map_levels=exp_levels, map_label='Exposure (s)',
                      title=('Combined exposure [' + det_filters[i] +
                             ' ]'))

    return ([exp_map_thin, exp_map_medium, exp_map_thick],
            [grid_centers_l, grid_centers_b], fig, axes)


def plot_fine_maps(folder, nh_data_file, pattern='*', telescope='XMM',
                   xmm_specfolder=None, csc_masterfile=None, min_exp=3,
                   legend=None):
    """Get fine resolution exposure maps of Chandra and XMM."""
    # Reading NH data
    nh_grid_centers, nh_grid = read_nhdata(nh_data_file)[1:]
    if telescope == 'Chandra':
        nh_grid_centers = [nh_grid_centers[0][32:47, 32:47],
                           nh_grid_centers[1][32:47, 32:47]]
        nh_grid = nh_grid[32:47, 32:47]
    print('NH data read')

    # Constructing exposure maps
    if telescope == 'Chandra':
        fine_grid = np.meshgrid(np.linspace(-0.56, 0.44, 121),
                                np.linspace(-0.56, 0.44, 121))
        exp_map, grid_centers = main_expmaps(
            folder, pattern, full_grid_l=fine_grid[0],
            full_grid_b=fine_grid[1], plot=False, telescope='Chandra')[:2]
        exp_maps = [exp_map]
        mos_exp_map = None
    else:
        fine_grid = np.meshgrid(np.linspace(-3, 3, 721),
                                np.linspace(-3, 3, 721))
        pn_exp_maps, grid_centers = main_expmaps(
            folder + '/pn_maps', pattern, full_grid_l=fine_grid[0],
            full_grid_b=fine_grid[1], plot=False, telescope='XMM')[:2]
        mos_exp_maps = main_expmaps(
            folder + '/mos_maps', pattern, full_grid_l=fine_grid[0],
            full_grid_b=fine_grid[1], plot=False, telescope='XMM')[0]
        exp_map = pn_exp_maps[0] + pn_exp_maps[1] + pn_exp_maps[2]
        mos_exp_map = mos_exp_maps[0] + mos_exp_maps[1] + mos_exp_maps[2]
        exp_maps = [pn_exp_maps, mos_exp_maps]
    print('Exp maps combined')

    # Read source coordinates
    if telescope == 'XMM':
        if xmm_specfolder is None:
            xmm_specfolder = '.'
        pn_files = glob2.glob(xmm_specfolder + '/*PN*grp1*cts.ds')
        mos_files = glob2.glob(xmm_specfolder + '/*MOS*grp1*cts.ds')
        srcs, coord = get_name_coord_exp_list(pn_files)[:-1]
        mos_srcs, mos_coord = get_name_coord_exp_list(mos_files)[:-1]
        mos_coord = mos_coord.copy().transpose()
        markers = ['o']
    else:
        csc_master = pd.read_table(csc_masterfile, header=13)
        srcs = np.array(csc_master['name'])
        csc_ra = np.array([
            (float(ra_str.split()[0]) + float(ra_str.split()[1])/60 +
             float(ra_str.split()[2])/3600) for ra_str in csc_master['ra']])*15
        csc_dec = np.array([
            (float(dec_str.split()[0]) - float(dec_str.split()[1])/60 -
             float(dec_str.split()[2])/3600) for dec_str in csc_master['dec']])
        csc_l, csc_b = eq_to_gal(csc_ra, csc_dec)
        csc_l[csc_l > 180] = csc_l[csc_l > 180] - 360
        coord = np.column_stack([csc_l, csc_b])
        mos_srcs = None
        mos_coord = None
        mos_exp_map = None
        markers = ['^', 'v', 'd']
    print('Coordinates read')

    # Plotting everything.
    plot_all(nh_grid_centers, nh_grid, exp_map, srcs, coord.transpose(),
             mos_exp_map, mos_srcs, mos_coord, min_exp=min_exp,
             exp_grid_cs=grid_centers, legend=legend, markers=markers)
    return ([nh_grid_centers, grid_centers], [nh_grid, exp_maps],
            [srcs, mos_srcs], [coord.transpose(), mos_coord])


def plot_exp_nhmaps(exp_map_folder, nh_data_file, telescope='XMM',
                    ind_plot=False, pattern=None, min_exp=3):
    """Plot exposure maps with nh contours."""
    nh_grid_coordinates, nh_grid_centers, nh_grid = read_nhdata(nh_data_file)
    nh_levels_fine = np.logspace(22.7, 24.0, 101)
    if pattern is None:
        if telescope == 'XMM':
            pattern = '*5000.FTZ'
        if telescope == 'Chandra':
            pattern = '*h_exp3.fits*'
    if telescope == 'XMM':
        (exp_maps, grid_centers, fig, axes) = main_expmaps(
            exp_map_folder, pattern=pattern,
            full_grid_l=nh_grid_coordinates[0],
            full_grid_b=nh_grid_coordinates[1], plot=ind_plot)
        if ind_plot:
            plot_contours(
                nh_grid_centers[0], nh_grid_centers[1], nh_grid, filled=True,
                axes=axes[1, 1], map_levels=nh_levels_fine, vmin=5.0E+22,
                map_label=r'N_H (cm^{-2})', title='Galactic Absorption')
        exp_map = exp_maps[0] + exp_maps[1] + exp_maps[2]
    elif telescope == 'Chandra':
        (exp_map, grid_centers, fig, axes) = main_expmaps(
            exp_map_folder, pattern=pattern,
            full_grid_l=nh_grid_coordinates[0],
            full_grid_b=nh_grid_coordinates[1], plot=ind_plot,
            telescope='Chandra')
        if ind_plot:
            plot_contours(
                nh_grid_centers[0], nh_grid_centers[1], nh_grid,
                filled=True, axes=axes[1, 1], map_levels=nh_levels_fine,
                vmin=5.0E+22, map_label=r'N_H (cm^{-2})',
                title='Galactic Absorption')
    nh_levels_coarse = np.logspace(22.7, 24.0, 5)
    exp_levels = np.logspace(min_exp, np.log10(np.max(exp_map)), 100)
    if telescope == 'Chandra':
        grid_centers = [grid_centers[0][32:47, 32:47],
                        grid_centers[1][32:47, 32:47]]
        nh_grid_centers = [nh_grid_centers[0][32:47, 32:47],
                           nh_grid_centers[1][32:47, 32:47]]
        exp_map = exp_map.copy()[32:47, 32:47]
        nh_grid = nh_grid.copy()[32:47, 32:47]
    fig, axes = plt.subplots()
    plot_contours(grid_centers[0], grid_centers[1], exp_map,
                  filled=True, axes=axes, map_levels=exp_levels,
                  map_label='Exposure (s)')
    plot_contours(nh_grid_centers[0], nh_grid_centers[1], nh_grid,
                  name_axes=False, axes=axes, map_levels=nh_levels_coarse)
    if telescope == 'XMM':
        return exp_maps, nh_grid, grid_centers, fig, axes
    if telescope == 'Chandra':
        return [exp_map], nh_grid, grid_centers, fig, axes
    return None, None, None, None, None


def plot_contours(grid_l, grid_b, map_image, name_axes=True, filled=False,
                  axes=None, cmap='YlOrRd_r', map_levels=None, vmin=1000,
                  map_label=None, title=None, lim_axes=False):
    """Plot contours."""
    if axes is None:
        fig, axes = plt.subplots()
        return_fig = True
    else:
        return_fig = False
    axes.set_title(title)
    if name_axes:
        axes.set_xlabel('Galactic Longitude')
        axes.set_ylabel('Galactic latitude')
    if lim_axes:
        axes.set_xlim(-2.0, 2.0)
        axes.set_ylim(-2.0, 2.0)
    if filled:
        map_cont = axes.contourf(grid_l, grid_b, map_image, levels=map_levels,
                                 norm=LogNorm(vmin=vmin), cmap=cmap)
        cbar = plt.colorbar(mappable=map_cont, ax=axes)
        cbar.set_label(map_label)
    else:
        contours = axes.contour(grid_l, grid_b, map_image, levels=map_levels,
                                colors='k')
        axes.clabel(contours, inline=True, fmt='%.0E')
        axes.legend()
    if return_fig:
        return fig, axes
    return None


def plot_all(grid_centers, nh_map, pn_exp_map, pn_srcs, pn_src_pos,
             mos_exp_map=None, mos_srcs=None, mos_src_pos=None,
             candidate_pn_srcs=None, candidate_mos_srcs=None, markers=None,
             color='#004488', legend=None, min_exp=3, exp_grid_cs=None):
    """Plot everything."""
    if mos_exp_map is not None:
        total_exp_map = pn_exp_map + 0.4*mos_exp_map
    else:
        total_exp_map = pn_exp_map.copy()
    if exp_grid_cs is None:
        exp_grid_cs = [grid_centers[0].copy(), grid_centers[1].copy()]
    if mos_srcs is not None:
        common_pn_bool = np.in1d(pn_srcs, mos_srcs)
        common_mos_bool = np.in1d(mos_srcs, pn_srcs)
    else:
        common_pn_bool = np.zeros(len(pn_srcs), dtype=bool)
    if candidate_pn_srcs is not None:
        candidate_pn_mask = np.in1d(pn_srcs, candidate_pn_srcs)
        only_pn_candidate_mask = np.logical_and(~common_pn_bool,
                                                candidate_pn_mask)
        if candidate_mos_srcs is not None:
            candidate_mos_mask = np.in1d(mos_srcs, candidate_mos_srcs)
            common_pn_candidate_mask = np.logical_and(common_pn_bool,
                                                      candidate_pn_mask)
            only_mos_candidate_mask = np.logical_and(~common_mos_bool,
                                                     candidate_mos_mask)
    if markers is None:
        markers = ['^', 'v', 'd']
    if legend is None:
        legend = ['Only PN detections', 'Only MOS detections',
                  'PN and MOS sources']
    exp_levels = np.logspace(min_exp, np.log10(np.max(total_exp_map)), 101)
    fig, axes = plot_contours(
        exp_grid_cs[0], exp_grid_cs[1], total_exp_map, filled=True,
        map_levels=exp_levels, map_label='Exposure (s)', vmin=10**min_exp)
    nh_levels = np.logspace(22.7, 24.0, 5)
    plot_contours(grid_centers[0], grid_centers[1], nh_map, name_axes=False,
                  axes=axes, map_levels=nh_levels, vmin=5.0E+22)
    axes.scatter(
        pn_src_pos[0][~common_pn_bool], pn_src_pos[1][~common_pn_bool],
        marker=markers[0], facecolor='none', edgecolor=color, label=legend[0])
    if candidate_pn_srcs is not None:
        axes.scatter(
            pn_src_pos[0][only_pn_candidate_mask],
            pn_src_pos[1][only_pn_candidate_mask], marker=markers[0],
            facecolor=color, edgecolor=color)
    if mos_srcs is not None:
        axes.scatter(
            mos_src_pos[0][~common_mos_bool], mos_src_pos[1][~common_mos_bool],
            marker=markers[1], facecolor='none', edgecolor=color,
            label=legend[1])
        axes.scatter(
            pn_src_pos[0][common_pn_bool], pn_src_pos[1][common_pn_bool],
            marker=markers[2], facecolor='none', edgecolor=color,
            label=legend[2])
        if candidate_mos_srcs is not None:
            axes.scatter(
                mos_src_pos[0][only_mos_candidate_mask],
                mos_src_pos[1][only_mos_candidate_mask], marker=markers[0],
                facecolor=color, edgecolor=color)
            axes.scatter(
                pn_src_pos[0][common_pn_candidate_mask],
                pn_src_pos[1][only_pn_candidate_mask], marker=markers[0],
                facecolor=color, edgecolor=color)
    axes.legend()
    return fig, axes


def write_pimmsfile(nh_value, exposure, count_limit=250, telescope='xmm',
                    detector='pn', det_filter=None,
                    pimms_file='pimms_flux_calc.xco'):
    """Write file to run with pimms."""
    cr_limit = count_limit/exposure
    if det_filter is None:
        if telescope == 'xmm':
            det_filter = 'medium'
        else:
            det_filter = ''
    with open(pimms_file, 'w') as writer:
        writer.writelines('model pl 2.0 ' + str(nh_value) + '\n')
        writer.writelines('from ' + telescope + ' ' + detector + ' ' +
                          det_filter + ' 2.0-10.0\n')
        writer.writelines('inst flux ergs 2.0-10.0 unabsorbed\n')
        writer.writelines('go ' + str(cr_limit) + '\n')
        writer.writelines('exit\n')


def run_pimmsfile(file='pimms_flux_calc.xco', outfile='flux_output.txt'):
    """Run the pimms file and return the unabsorbed flux."""
    os.system('/Users/pavanrh/bin_astro/pimms4_13a/pimms @' + file + ' > ' +
              outfile)
    with open(outfile, 'r') as reader:
        for line in reader:
            words = line.split()
            if words[2] == 'predicts':
                unabs_flux = float(words[-2])
                break

    return unabs_flux


def get_lim_flux(exp_map, nh_map, telescope='XMM', detector='pn',
                 det_filter=None):
    """Get limiting flux."""
    if det_filter is None and telescope == 'XMM':
        det_filter = 'medium'
    lim_flux_grid = np.zeros_like(nh_map)
    num_rows, num_cols = nh_map.shape
    for i in range(num_rows):
        for j in range(num_cols):
            if nh_map[i, j] >= 5.0E+22 and exp_map[i, j] >= 1000:
                write_pimmsfile(nh_map[i, j], exp_map[i, j],
                                telescope=telescope, detector=detector,
                                det_filter=det_filter)
                lim_flux_grid[i, j] = run_pimmsfile()
            else:
                lim_flux_grid[i, j] = np.nan
    return lim_flux_grid


def get_lim_flux_list(exp_maps, nh_map, telescope='XMM', detector='pn'):
    """Get limiting flux of a list."""
    lim_flux_list = []
    if telescope == 'XMM':
        filter_list = ['thin', 'medium', 'thick']
    else:
        filter_list = [None]
    for i, exp_map in enumerate(exp_maps):
        if np.max(exp_map) > 1000:
            lim_flux_grid = get_lim_flux(exp_map, nh_map, telescope, detector,
                                         filter_list[i])
            lim_flux_list.append(lim_flux_grid)
        else:
            lim_flux_list.append(np.ones_like(exp_map)*np.nan)
    # Lim flux of the combined spectra is similar to parallel resistors
    lim_flux_combined_inv = np.zeros_like(nh_map)
    for lim_flux in lim_flux_list:
        lim_flux[np.isnan(lim_flux)] = np.inf
        lim_flux_combined_inv += 1.0/lim_flux
    lim_flux_combined_inv[lim_flux_combined_inv == 0] = np.nan
    return 1.0/lim_flux_combined_inv, lim_flux_list


def plot_lim_flux(grid, lim_flux_map):
    """Plot limitting flux."""
    min_flux = np.min(lim_flux_map[~np.isnan(lim_flux_map)])
    max_flux = np.max(lim_flux_map[~np.isnan(lim_flux_map)])
    flux_levels = np.logspace(np.log10(min_flux), np.log10(max_flux), 101)
    plot_contours(grid[0], grid[1], lim_flux_map, filled=True,
                  map_levels=flux_levels, vmin=min_flux,
                  map_label=r'Limiting flux (ergs s$^{-1}$ cm$^{-2})')
    plt.tight_layout()


def estimate_agn(lim_flux_map, interp_fn, grid_area):
    """"Estimate the number of AGN from limiting flux map."""
    n_s14 = 10**interp_fn(np.log10(lim_flux_map))
    agn_perbin = n_s14/(lim_flux_map/1.0E-14)**1.5*grid_area
    return np.sum(agn_perbin[~np.isnan(agn_perbin)])


def main():
    """Main function."""
    # XMM telescope
    agn_interp_data = np.loadtxt('/Users/pavanrh/Documents/UofA_projects/' +
                                 'GalacticBulge/Total_AGN_flux.xls',
                                 skiprows=1)
    agn_interp = interpolate.CubicSpline(np.log10(agn_interp_data[:, 0]),
                                         np.log10(agn_interp_data[:, 1]))
    nh_file = ('/Volumes/Pavan_Work_SSD/GalacticBulge_4XMM_Chandra/' +
               'data/gal_coords_NH.csv')
    exp_map_folders = [('/Volumes/Pavan_Work_SSD/GalacticBulge_4XMM_Chandra/' +
                        'data/xmm_expmaps/pn_maps/'),
                       ('/Volumes/Pavan_Work_SSD/GalacticBulge_4XMM_Chandra/' +
                        'data/xmm_expmaps/mos_maps/'),
                       '/Volumes/Pavan_Work_SSD/Test_folder']
    pattern_list = ['*5000.FTZ', '*5000.FTZ', 'sgr*']
    telescope_arr = ['XMM', 'XMM', 'Chandra']
    detector_arr = ['pn', 'mos', 'acis-i']
    min_exps = [3, 3, 4]
    exp_maps_arr = []
    fig_arr = []
    axes_arr = []
    lim_flux_total_arr = []
    lim_flux_perfilter_arr = []
    agn_arr = []
    for i, folder in enumerate(exp_map_folders):
        exp_maps, nh_grid, grid_centers, fig, axes = plot_exp_nhmaps(
            exp_map_folder=folder, nh_data_file=nh_file,
            telescope=telescope_arr[i], pattern=pattern_list[i],
            min_exp=min_exps[i])
        if i == 0:
            nh_grid_xmm = nh_grid.copy()
            grid_centers_xmm = copy.copy(grid_centers)
        exp_maps_arr.append(exp_maps)
        fig_arr.append(fig)
        axes_arr.append(axes)

        lim_flux_total, lim_flux_perfilter = get_lim_flux_list(
            exp_maps, nh_grid, telescope=telescope_arr[i],
            detector=detector_arr[i])
        plot_lim_flux(grid_centers, lim_flux_total)
        lim_flux_total_arr.append(lim_flux_total)
        lim_flux_perfilter_arr.append(lim_flux_perfilter)

        num_agn = estimate_agn(lim_flux_total, agn_interp, (6.0/79.0)**2)
        agn_arr.append(num_agn)

    return (nh_grid_xmm, grid_centers_xmm, exp_maps_arr, lim_flux_total_arr,
            lim_flux_perfilter_arr, agn_arr)
