"""Python program to estimate the number of total MSPs."""

import numpy as np
import matplotlib.pyplot as plt


def get_grid_corners(lgrid_centers, bgrid_centers):
    """Get the corners of the grid from the central values."""
    lgrid_1d = lgrid_centers[0]
    bgrid_1d = bgrid_centers[:, 0]
    l_grid_corners_1d = np.hstack(
        [1.5*lgrid_1d[0] - 0.5*lgrid_1d[1],
         0.5*(lgrid_1d[1:] + lgrid_1d[:-1]),
         1.5*lgrid_1d[-1] - 0.5*lgrid_1d[-2]])
    b_grid_corners_1d = np.hstack(
        [1.5*bgrid_1d[0] - 0.5*bgrid_1d[1],
         0.5*(bgrid_1d[1:] + bgrid_1d[:-1]),
         1.5*bgrid_1d[-1] - 0.5*bgrid_1d[-2]])
    return np.meshgrid(l_grid_corners_1d, b_grid_corners_1d)


def load_grid_centers(lgrid_file, bgrid_file):
    """Load grid centers of the limiting flux map."""
    lgrid_centers = np.loadtxt(lgrid_file)
    bgrid_centers = np.loadtxt(bgrid_file)
    grid_corners = get_grid_corners(lgrid_centers, bgrid_centers)
    return grid_corners, [lgrid_centers, bgrid_centers]