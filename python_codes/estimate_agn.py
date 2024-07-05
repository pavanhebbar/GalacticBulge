"""Python program to estimate the number of background AGN.

Calculate in galactic coordinates and convert to equatorial coordinates.
"""

import glob2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS


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
    gal_l_2d = gal_l_vals.reshape(num_grids, num_grids) # Transp. of convention
    gal_b_2d = gal_b_vals.reshape(num_grids, num_grids) # Transp. of convention
    gal_l_1d_grid = np.hstack(
        [gal_l_2d[0, 0] - 0.5*(gal_l_2d[1, 0] - gal_l_2d[0, 0]),
         0.5*(gal_l_2d[1:, 0] + gal_l_2d[:-1, 0]),
         gal_l_2d[-1, 0] + 0.5*(gal_l_2d[-1, 0] - gal_l_2d[-2, 0])])
    gal_b_1d_grid = np.hstack(
        [gal_b_2d[0, 0] - 0.5*(gal_b_2d[0, 1] - gal_b_2d[0, 0]),
         0.5*(gal_b_2d[0, 1:] + gal_b_2d[0, :-1]),
         gal_b_2d[0, -1] + 0.5*(gal_b_2d[0, -1] - gal_b_2d[0, -2])])
    nh_grid = nh_vals.reshape(num_grids, num_grids)
    return np.meshgrid(gal_l_1d_grid, gal_b_1d_grid), nh_grid.transpose()


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


def add_expmap(expmap_file, full_grid_l, full_grid_b, full_exp_map):
    """Add exposure map to the main grid."""
    hdu_table = fits.open(expmap_file)
    exp_header = hdu_table[0].header
    exp_map = hdu_table[0].data
    exp_map_coord = wcs_to_gal_grid(WCS(exp_header))
    gal_l_coordinates = exp_map_coord[0].reshape((-1, 1))
    gal_b_coordinates = exp_map_coord[1].reshape((-1, 1))
    l_indices, b_indices = get_grid_indices(
        full_grid_l, full_grid_b,
        np.hstack([gal_l_coordinates, gal_b_coordinates]))
    for i in range(full_grid_l.shape[0]):
        for j in range(full_grid_l.shape[1]):
            bin_indices = np.where(np.logical_and(
                l_indices == i, b_indices == j))[0]
            if len(bin_indices) > 0:
                full_exp_map[i, j] += np.mean(exp_map[bin_indices])
    return full_exp_map

def combine_exp_maps(exp_map_list, full_grid_l, full_grid_b):
    """Add exposure maps."""
    exp_map_final = np.zeros_like(full_grid_l)
    for exp_map in exp_map_list:
        exp_map_final = add_expmap(exp_map, full_grid_l, full_grid_b,
                                   exp_map_final.copy())
    return exp_map_final, full_grid_l, full_grid_b


def plot_exp_nh(exp_grids, exp_map, nh_grids, nh_maps):
    """Plot exposure and Nh maps.

    Exposures will be plotted in colors and NH contours will be shown on them.
    """
    plt.figure()
    plt.contourf(exp_grids[0], exp_grids[1], exp_map, cmap='inferno',
                 origin='lower')



def main(folder, outfile='Combined_exposures.ftz', pattern='*'):
    """Main function."""
    exp_map_list = glob2.glob('')
