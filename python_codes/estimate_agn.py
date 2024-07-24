"""Python program to estimate the number of background AGN.

Calculate in galactic coordinates and convert to equatorial coordinates.
"""

from tkinter import N
import glob2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from matplotlib.colors import LogNorm


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


def get_coord_exp_list(file_list):
    """Get eposures and coordinates of a list of spectra."""
    coord_list = np.zeros((len(file_list), 2), dtype=float)
    exp_list = np.zeros(len(file_list), dtype=float)
    for i, file in enumerate(file_list):
        coord_list[i], exp_list[i] = get_coordinates_exposure(file)

    return coord_list, exp_list


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


'''
def add_expmap(expmap_file, full_grid_l, full_grid_b, full_exp_map):
    """Add exposure map to the main grid."""
    hdu_table = fits.open(expmap_file)
    exp_map = hdu_table[0].data
    exp_map_coord = wcs_to_gal_grid(WCS(hdu_table[0].header))
    gal_l_coordinates = exp_map_coord[0].reshape((-1, 1))
    gal_b_coordinates = exp_map_coord[1].reshape((-1, 1))
    l_indices, b_indices = get_grid_indices(
        full_grid_l, full_grid_b,
        np.hstack([gal_l_coordinates, gal_b_coordinates]))
    grid_count = np.zeros_like(full_exp_map)
    exp_map_grid = np.zeros_like(full_exp_map)
    for i, l_index in enumerate(l_indices):
        grid_count[b_indices[i], l_index] += 1
        exp_map_grid[b_indices[i], l_index] += exp_map.reshape(-1)[i]
    full_exp_map += (exp_map_grid/grid_count)
    return full_exp_map
'''


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
                      map_levels=exp_levels, map_label='Exposure (ks)',
                      title=('Combined exposure [' + det_filters[i] +
                             ' ]'))

    return ([exp_map_thin, exp_map_medium, exp_map_thick],
            [grid_centers_l, grid_centers_b], fig, axes)


def plot_exp_nhmaps(exp_map_folder, nh_data_file, telescope='XMM',
                    ind_plot=False, pattern=None):
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
    exp_levels = np.logspace(3, 6.3, 100)
    fig, axes = plt.subplots()
    plot_contours(grid_centers[0], grid_centers[1], exp_map,
                  filled=True, axes=axes, map_levels=exp_levels,
                  map_label='Exposure (s)')
    plot_contours(nh_grid_centers[0], nh_grid_centers[1], nh_grid,
                  name_axes=False, axes=axes, map_levels=nh_levels_coarse)
    if telescope == 'XMM':
        return exp_maps, nh_grid, grid_centers, fig, axes
    if telescope == 'Chandra':
        return exp_map, nh_grid, grid_centers, fig, axes


def plot_contours(grid_l, grid_b, map_image, name_axes=True, filled=False,
                  axes=None, cmap='YlOrRd_r', map_levels=None, vmin=1000,
                  map_label=None, title=None, lim_axes=False):
    """Plot contours."""
    if axes is None:
        fig, axes = plt.subplots()
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
    return fig, axes


def plot_all(grid_centers, nh_map, pn_exp_map, pn_srcs, pn_src_pos,
             mos_exp_map=None, mos_srcs=None, mos_src_pos=None,
             candidate_pn_srcs=None, candidate_mos_srcs=None, markers=None,
             color='#004488', legend=None):
    """Plot everything."""
    if mos_exp_map is not None:
        pn_exp_map += 0.4*mos_exp_map
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
        legend = ['PN sources', 'MOS sources', 'PN and MOS sources']
    exp_levels = np.logspace(3, np.log10(np.max(pn_exp_map)), 101)
    fig, axes = plot_contours(
        grid_centers[0], grid_centers[1], pn_exp_map, filled=True,
        map_levels=exp_levels, map_label='Exposure')
    nh_levels = np.logspace(22.7, 24.0, 5)
    plot_contours(grid_centers[0], grid_centers[1], nh_map, name_axes=False,
                  axes=axes, map_levels=nh_levels, vmin=5.0E+22)
    axes.scatter(
        pn_src_pos[0][~common_pn_bool], pn_src_pos[1][~common_pn_bool],
        marker=markers[0], facecolor='none', edgecolor=color)
    if candidate_pn_mask:
        axes.scatter(
            pn_src_pos[0][only_pn_candidate_mask],
            pn_src_pos[1][only_pn_candidate_mask], marker=markers[0],
            facecolor=color, edgecolor=color)
    if mos_srcs is not None:
        axes.scatter(
            mos_src_pos[0][~common_mos_bool], mos_src_pos[1][~common_mos_bool],
            marker=markers[1], facecolor='none', edgecolor=color)
        axes.scatter(
            pn_src_pos[0][common_pn_bool], pn_src_pos[1][common_pn_bool],
            marker=markers[2], facecolor='none', edgecolor=color)
        if candidate_mos_mask:
             axes.scatter(
                mos_src_pos[0][only_mos_candidate_mask],
                mos_src_pos[1][only_mos_candidate_mask], marker=markers[0],
                facecolor=color, edgecolor=color)
             axes.scatter(
                pn_src_pos[0][common_pn_candidate_mask],
                pn_src_pos[1][only_pn_candidate_mask], marker=markers[0],
                facecolor=color, edgecolor=color)
    

def write_pimmsfile(nh, exposure, count_limit=250, telescope='xmm',
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
        writer.writelines('model pl 2.0 ' + str(nh) + '\n')
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


def match_exp_nh_maps(exp_map, nh_map, exp_grid_centers):
    """Match the sizes/resolution of exposure and nh_maps."""
    pass


def get_lim_flux(exp_map, nh_map, telescope='XMM', detector='pn',
                 det_filter='medium'):
    """Get limiting flux."""
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
    filter_list = ['thin', 'medium', 'thick']
    for i, exp_map in enumerate(exp_maps):
        if np.max(exp_map) > 1000:
            lim_flux_grid = get_lim_flux(exp_map, nh_map, telescope, detector,
                                         filter_list[i])
            lim_flux_list.append(lim_flux_grid)
        else:
            lim_flux_list.append(np.ones_like(exp_map)*np.nan)
    return lim_flux_list


def estimate_agn(lim_flux_list, n_s14, grid_area):
    """"Estimate the number of AGN from limiting flux map."""
    num_agn_all = 0
    for lim_flux_map in lim_flux_list:
        num_agn_perbin = n_s14/(lim_flux_map/1.0E-14)**1.5*grid_area
        num_agn_all += np.sum(num_agn_perbin[~np.isnan(num_agn_perbin)])
    return num_agn_all


def main():
    """Main function."""
    # XMM telescope
    xmm_pn_exp_maps, xmm_nh_grid = plot_exp_nhmaps(
        exp_map_folder=('/Volumes/Pavan_Work_SSD/GalacticBulge_4XMM_Chandra/' +
                        'data/xmm_expmaps/pn_maps/'),
        nh_data_file=('/Volumes/Pavan_Work_SSD/GalacticBulge_4XMM_Chandra/' +
                      'data/gal_coords_NH.csv'))
    xmm_lim_flux_list = []
    xmm_filters
    for exp_map in xmm_exp_maps:
        if np.max(exp_map) > 1000:
            lim_flux_grid = 
    chandra_exp_maps, chandra_nh_grid = plot_exp_nhmaps(
        exp_map_folder='/Volumes/Pavan_work/chandra_deep2/all_expmaps_1ks/',
        nh_data_file=('/Volumes/Pavan_Work_SSD/GalacticBulge_4XMM_Chandra/' +
                      'data/gal_coords_NH.csv'),
        telescope='Chandra')  # Will take lots of time
