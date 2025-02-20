from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm, ZScaleInterval, ImageNormalize
from astropy.io import fits
import cmcrameri.cm as cmc
import numpy as np
import torch

def plot_wcs_comparison(source_hdu: fits.PrimaryHDU, target_hdu: fits.PrimaryHDU):
    """
    Create a visual comparison of two astronomical images with different WCS (World Coordinate System) projections.

    This function generates a three-panel figure that shows:
    1. The source image with coordinate grid
    2. The target image with coordinate grid
    3. An overlay of both WCS grids to visualize the coordinate transformation

    The visualization helps to understand the differences between two WCS projections
    and the transformations required for reprojection.

    Parameters
    ----------
    source_hdu : fits.PrimaryHDU
        The source image HDU containing both the image data and WCS information
        in its header. This is the "before" image.

    target_hdu : fits.PrimaryHDU
        The target image HDU containing both the image data and WCS information
        in its header. This is the "after" image or the reference frame.

    Returns
    -------
    None
        This function does not return any value but displays a matplotlib figure
        with the three-panel comparison.

    Notes
    -----
    The function uses:
    - WCS from astropy for coordinate handling
    - simple_norm from astropy.visualization for image normalization
    - A preset RA/Dec grid covering 149.8-150.3 degrees in RA and 29.8-30.3 degrees in Dec
    - The 'lipari' colormap from cmc
    - Asinh scaling for better visualization of astronomical data

    The third panel shows how the same celestial coordinates map to different
    pixel positions in the two images, which is useful for understanding the
    distortions introduced by different projections.

    Examples
    --------
    >>> from astropy.io import fits
    >>> # Load source and target images
    >>> source = fits.open('original_image.fits')[0]
    >>> target = fits.open('reprojected_image.fits')[0]
    >>> # Create the comparison plot
    >>> plot_wcs_comparison(source, target)

    See Also
    --------
    WCS : Astropy World Coordinate System class


    """
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
    Create a side-by-side comparison of original and reprojected astronomical images.

    This function generates a two-panel figure displaying the original source image
    alongside its reprojected version for visual comparison. The images are displayed
    with the same normalization to facilitate direct comparison of features and flux.

    Parameters
    ----------
    source_hdu : fits.PrimaryHDU or fits.ImageHDU
        The original source image HDU containing image data. This represents the
        input image before reprojection.

    reprojected_source : numpy.ndarray or torch.Tensor
        The reprojected version of the source image. Can be either a NumPy array
        or a PyTorch tensor (which will be converted to NumPy for plotting).

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the side-by-side comparison plot.
        This allows for further customization or saving the figure.

    Notes
    -----
    This function:
    - Automatically converts PyTorch tensors to NumPy arrays if needed
    - Uses ZScale normalization from astropy for optimal visualization of astronomical data
    - Applies the same normalization to both images for fair comparison
    - Shows image dimensions in the subplot titles
    - Uses the 'lipari' colormap which is effective for astronomical imagery
    - Displays images with 'lower' origin, following astronomical convention

    Examples
    --------
    >>> from astropy.io import fits
    >>> from reprojection import calculate_reprojection
    >>>
    >>> # Load source image
    >>> source_hdu = fits.open('source_image.fits')[0]
    >>>
    >>> # Get reprojected image (example with some target WCS)
    >>> reprojected_data, _ = calculate_reprojection(source_hdu, target_wcs)
    >>>
    >>> # Compare the original and reprojected images
    >>> fig = compare_images(source_hdu, reprojected_data)
    >>> fig.savefig('comparison.png', dpi=300)
    """
    # Convert tensor back to CPU numpy if needed
    if torch.is_tensor(reprojected_source):
        reprojected_source = reprojected_source.cpu().numpy()

    source_data = source_hdu.data

    # Create normalization based on both images
    zscale = ZScaleInterval()
    norm_source = ImageNormalize(source_data, interval=zscale)
    norm_reproj = ImageNormalize(source_data, interval=zscale)

    # Create the figure with just two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot original
    im1 = ax1.imshow(source_data, norm=norm_source, origin='lower', cmap=cmc.lipari)
    ax1.set_title(f'Original Source\n{source_data.shape}')
    plt.colorbar(im1, ax=ax1)

    # Plot reprojected
    im2 = ax2.imshow(reprojected_source, norm=norm_reproj, origin='lower', cmap=cmc.lipari)
    ax2.set_title(f'Reprojected Source\n{reprojected_source.shape}')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    return fig
