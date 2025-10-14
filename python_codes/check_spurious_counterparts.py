"""Python program to estimate the rate of spurious counterparts."""


import numpy as np


def shift_source(source_coord, shift_radius):
    """Shift_source position.

    source coord and shift radius in deg. Only apply for small shifts
    """
    ra_src, dec_src = source_coord
    if isinstance(ra_src, float):
        shift_angle = np.random.uniform(0, 2 * np.pi)
    else:
        shift_angle = np.random.uniform(0, 2 * np.pi, size=len(ra_src))
    new_ra = (ra_src +
              shift_radius * np.sin(shift_angle) / np.cos(dec_src*np.pi/180.))
    new_dec = dec_src + shift_radius * np.cos(shift_angle)
    return [new_ra, new_dec]


def calc_dist(source_coord1, source_coord2):
    """Calculate distance between two sources.

    All values are in degrees
    """
    ra_1, dec_1 = source_coord1
    ra_2, dec_2 = source_coord2
    dist = np.arccos(
        np.sin(dec_1 * np.pi/180.) * np.sin(dec_2*np.pi/180.) +
        np.cos(dec_1 * np.pi/180.) * np.cos(dec_2*np.pi/180.) * np.cos(
            (ra_2 - ra_1)*np.pi/180.))*180/np.pi
    return dist


def get_numcounterparts(coords, ref_coord, max_offset):
    """Get the number of counterparts of the source given maximum offset."""
    dists = calc_dist(coords.transpose(), ref_coord)
    return len(np.where(dists < max_offset)[0])


def get_ncounterparts_fullcatalog(coords, ref_coords, max_offset):
    """Get number of counterparts for all sources in the counterparts"""
    num_counterparts_arr = np.zeros(len(ref_coords), dtype=int)
    for i, src_coord in enumerate(ref_coords):
        num_counterparts_arr[i] = get_numcounterparts(coords, src_coord,
                                                      max_offset)

    return num_counterparts_arr, np.sum(num_counterparts_arr)


def compare_cats_givenoffset(coords_ref, coords_compare, shift_dist, offset,
                             num_sim=100):
    """"Compare two catalogs given the offset."""
    num_counterparts_all_arr = np.zeros(num_sim, dtype=int)
    for i in range(num_sim):
        coords_ref_shifted = shift_source(coords_ref, shift_dist)
        num_counterparts_all_arr[i] = get_ncounterparts_fullcatalog(
            coords_compare, coords_ref_shifted, offset)[1]

    return np.mean(num_counterparts_all_arr), np.std(num_counterparts_all_arr)


def compare_cats_varoffset(coords_ref, coords_compare, shift_dist,
                           num_sim=100):
    """Compare how the number of random counterparts changes with offset"""
    offsets = np.linspace(0.1, 2.0, 20)
    ncounterparts_offset_mean = np.zeros(20, dtype=float)
    ncounterparts_offset_std = np.zeros(20, dtype=float)
    for i, offset in enumerate(offsets):
        ncounterparts_offset_mean[i], ncounterparts_offset_std = (
            compare_cats_givenoffset(coords_ref, coords_compare, shift_dist,
                                     offset, num_sim))

    return ncounterparts_offset_mean, ncounterparts_offset_std
