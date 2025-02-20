from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm, ZScaleInterval, ImageNormalize
from astropy.io import fits
import cmcrameri.cm as cmc
import numpy as np
import torch

def plot_wcs_comparison(source_hdu: fits.PrimaryHDU, target_hdu: fits.PrimaryHDU):
    # Create WCS objects
    source_wcs = WCS(source_hdu.header)
    target_wcs = WCS(target_hdu.header)

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))

    # Plot source image
    ax1 = fig.add_subplot(131, projection=source_wcs)
    norm = simple_norm(source_hdu.data, 'asinh')
    im1 = ax1.imshow(source_hdu.data, norm=norm, cmap=cmc.lipari)
    ax1.grid(color='white', ls='solid', alpha=0.5)
    ax1.set_title('Source Image')

    # Plot target image
    ax2 = fig.add_subplot(132, projection=target_wcs)
    norm = simple_norm(target_hdu.data, 'asinh')
    im2 = ax2.imshow(target_hdu.data, norm=norm, cmap=cmc.lipari)
    ax2.grid(color='white', ls='solid', alpha=0.5)
    ax2.set_title('Target Image')

    # Plot overlay of WCS grids
    ax3 = fig.add_subplot(133)

    # Plot coordinate grids for both images
    ra_grid = np.linspace(149.8, 150.3, 10)
    dec_grid = np.linspace(29.8, 30.3, 10)

    # Plot source grid
    for ra in ra_grid:
        dec = np.linspace(29.8, 30.3, 100)
        ra_arr = np.full_like(dec, ra)
        pixels = source_wcs.wcs_world2pix(np.column_stack((ra_arr, dec)), 0)
        ax3.plot(pixels[:, 0], pixels[:, 1], 'b-', alpha=0.5, label='Source' if ra == ra_grid[0] else '')

    for dec in dec_grid:
        ra = np.linspace(149.8, 150.3, 100)
        dec_arr = np.full_like(ra, dec)
        pixels = source_wcs.wcs_world2pix(np.column_stack((ra, dec_arr)), 0)
        ax3.plot(pixels[:, 0], pixels[:, 1], 'b-', alpha=0.5)

    # Plot target grid
    for ra in ra_grid:
        dec = np.linspace(29.8, 30.3, 100)
        ra_arr = np.full_like(dec, ra)
        pixels = target_wcs.wcs_world2pix(np.column_stack((ra_arr, dec)), 0)
        ax3.plot(pixels[:, 0], pixels[:, 1], 'r-', alpha=0.5, label='Target' if ra == ra_grid[0] else '')

    for dec in dec_grid:
        ra = np.linspace(149.8, 150.3, 100)
        dec_arr = np.full_like(ra, dec)
        pixels = target_wcs.wcs_world2pix(np.column_stack((ra, dec_arr)), 0)
        ax3.plot(pixels[:, 0], pixels[:, 1], 'r-', alpha=0.5)

    ax3.set_title('WCS Grid Overlay')
    ax3.legend()
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.show()


def compare_images(source_hdu, reprojected_source):
    """
    Compare original and reprojected images
    """
    # Convert tensor back to CPU numpy if needed
    if torch.is_tensor(reprojected_source):
        reprojected_source = reprojected_source.cpu().numpy()

    source_data = source_hdu.data

    # Create normalization based on both images
    zscale = ZScaleInterval()
    norm_source = ImageNormalize(source_data, interval=zscale)
    norm_reproj = ImageNormalize(reprojected_source, interval=zscale)

    # Create the figure with just two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot original
    im1 = ax1.imshow(source_data, norm=norm_source, origin='lower', cmap='viridis')
    ax1.set_title(f'Original Source\n{source_data.shape}')
    plt.colorbar(im1, ax=ax1)

    # Plot reprojected
    im2 = ax2.imshow(reprojected_source, norm=norm_reproj, origin='lower', cmap='viridis')
    ax2.set_title(f'Reprojected Source\n{reprojected_source.shape}')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    return fig
