from typing import Tuple

import astropy
import numpy as np
import torch
from astropy.io import fits

from reprojection.utils import get_device
from astropy.io.fits import Header, PrimaryHDU
from astropy.wcs import WCS
EPSILON = 1e-10


@torch.jit.script
def interpolate_image(source_image: torch.Tensor, grid: torch.Tensor, interpolation_mode: str) -> torch.Tensor:
    """JIT-compiled image interpolation using grid_sample"""
    return torch.nn.functional.grid_sample(
        source_image,
        grid,
        mode=interpolation_mode,
        align_corners=True,
        padding_mode='zeros'
    )


def get_sip_coeffs(wcs: astropy.wcs.WCS) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract SIP polynomial coefficients from a WCS object.

    Parameters:
    -----------
    wcs : astropy.wcs.WCS
        WCS object potentially containing SIP distortion

    Returns:
    --------
    dict:
        Dictionary containing SIP coefficient matrices A, B, AP, BP and orders
    """
    sip_coeffs = {}

    # Check if SIP distortion is present
    sip = getattr(wcs.wcs, 'sip', None)
    if sip is None:
        return None

    # Extract the SIP matrices
    sip_coeffs['a_order'] = sip.a_order
    sip_coeffs['b_order'] = sip.b_order
    sip_coeffs['a'] = sip.a
    sip_coeffs['b'] = sip.b

    # Check for inverse coefficients
    if hasattr(sip, 'ap_order') and sip.ap_order > 0:
        sip_coeffs['ap_order'] = sip.ap_order
        sip_coeffs['ap'] = sip.ap
    else:
        sip_coeffs['ap_order'] = 0

    if hasattr(sip, 'bp_order') and sip.bp_order > 0:
        sip_coeffs['bp_order'] = sip.bp_order
        sip_coeffs['bp'] = sip.bp
    else:
        sip_coeffs['bp_order'] = 0

    return sip_coeffs


def apply_sip_distortion(u: torch.Tensor, v: torch.Tensor, sip_coeffs: Tuple, device: str='cpu'):
    """
    Apply SIP distortion to intermediate pixel coordinates.

    Parameters:
    -----------
    u, v : torch.Tensor
        Intermediate pixel coordinates (before distortion)
    sip_coeffs : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        SIP coefficient matrices
    device : torch.device, optional
        Device to place tensors on

    Returns:
    --------
    tuple:
        (u', v') distorted coordinates
    """
    if sip_coeffs is None:
        return u, v

    # Convert to tensors if needed
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, device=device)
        v = torch.tensor(v, device=device)

    # Get the SIP orders
    a_order = sip_coeffs['a_order']

    # Convert coefficient matrices to tensors
    a_matrix = torch.tensor(sip_coeffs['a'], device=device)
    b_matrix = torch.tensor(sip_coeffs['b'], device=device)

    # Initialize correction terms
    f_u = torch.zeros_like(u)
    f_v = torch.zeros_like(v)

    # For array inputs, reshape to make computation easier
    orig_shape = u.shape
    u_flat = u.view(-1) if u.dim() > 0 else u.unsqueeze(0)
    v_flat = v.view(-1) if v.dim() > 0 else v.unsqueeze(0)
    f_u_flat = f_u.view(-1) if f_u.dim() > 0 else f_u.unsqueeze(0)
    f_v_flat = f_v.view(-1) if f_v.dim() > 0 else f_v.unsqueeze(0)

    # Apply the polynomial distortion
    for i in range(a_order + 1):
        for j in range(a_order + 1 - i):
            if i == 0 and j == 0:
                continue  # Skip the 0,0 term

            # Compute u^i * v^j for all points
            pow_term = (u_flat ** i) * (v_flat ** j)

            # Apply coefficient
            f_u_flat += a_matrix[i, j] * pow_term
            f_v_flat += b_matrix[i, j] * pow_term

    # Reshape back to original shape if needed
    if u.dim() > 0:
        f_u = f_u_flat.view(orig_shape)
        f_v = f_v_flat.view(orig_shape)
    else:
        f_u = f_u_flat[0]
        f_v = f_v_flat[0]

    # Add the distortion terms to get the corrected coordinates
    u_corrected = u + f_u
    v_corrected = v + f_v

    return u_corrected, v_corrected


def apply_inverse_sip_distortion(u: torch.Tensor, v:torch.Tensor, sip_coeffs:Tuple, device: str='cpu'):
    """
    Apply inverse SIP distortion to go from distorted to intermediate coordinates.

    Parameters:
    -----------
    u, v : torch.Tensor
        Distorted coordinates
    sip_coeffs : Tuple
        SIP coefficient matrices
    device : torch.device, optional
        Device to place tensors on

    Returns:
    --------
    tuple:
        (u', v') undistorted coordinates
    """
    if sip_coeffs is None:
        return u, v

    # Check if inverse coefficients are available
    if sip_coeffs['ap_order'] == 0 or sip_coeffs['bp_order'] == 0:
        # Use iterative method if inverse coefficients aren't available
        return iterative_inverse_sip(u, v, sip_coeffs, device)

    # Convert to tensors if needed
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, device=device)
        v = torch.tensor(v, device=device)

    # Get the SIP orders
    ap_order = sip_coeffs['ap_order']

    # Convert coefficient matrices to tensors
    ap_matrix = torch.tensor(sip_coeffs['ap'], device=device)
    bp_matrix = torch.tensor(sip_coeffs['bp'], device=device)

    # Initialize correction terms
    f_u = torch.zeros_like(u)
    f_v = torch.zeros_like(v)

    # For array inputs, reshape to make computation easier
    orig_shape = u.shape
    u_flat = u.view(-1) if u.dim() > 0 else u.unsqueeze(0)
    v_flat = v.view(-1) if v.dim() > 0 else v.unsqueeze(0)
    f_u_flat = f_u.view(-1) if f_u.dim() > 0 else f_u.unsqueeze(0)
    f_v_flat = f_v.view(-1) if f_v.dim() > 0 else f_v.unsqueeze(0)

    # Apply the polynomial correction
    for i in range(ap_order + 1):
        for j in range(ap_order + 1 - i):
            if i == 0 and j == 0:
                continue  # Skip the 0,0 term

            # Compute u^i * v^j for all points
            pow_term = (u_flat ** i) * (v_flat ** j)

            # Apply coefficient
            f_u_flat += ap_matrix[i, j] * pow_term
            f_v_flat += bp_matrix[i, j] * pow_term

    # Reshape back to original shape if needed
    if u.dim() > 0:
        f_u = f_u_flat.view(orig_shape)
        f_v = f_v_flat.view(orig_shape)
    else:
        f_u = f_u_flat[0]
        f_v = f_v_flat[0]

    # Add the correction terms to get the undistorted coordinates
    u_corrected = u + f_u
    v_corrected = v + f_v

    return u_corrected, v_corrected

def iterative_inverse_sip(u: torch.Tensor, v: torch.Tensor, sip_coeffs: Tuple, device: str='cpu', max_iter: int=20, tol: float=1e-8):
    """
    Iteratively solve for undistorted coordinates when inverse SIP coefficients
    are not available.

    Parameters:
    -----------
    u, v : torch.Tensor
        Distorted coordinates
    sip_coeffs : Tuple
        SIP coefficient matrices
    device : torch.device, optional
        Device to place tensors on
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Convergence tolerance

    Returns:
    --------
    tuple:
        (u', v') undistorted coordinates
    """
    # Convert to tensors if needed
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, device=device)
        v = torch.tensor(v, device=device)

    # Initial guess: undistorted = distorted
    u_undist = u.clone()
    v_undist = v.clone()

    for i in range(max_iter):
        # Apply forward SIP to get predicted distorted coordinates
        u_pred, v_pred = apply_sip_distortion(u_undist, v_undist, sip_coeffs, device)

        # Compute error
        u_error = u - u_pred
        v_error = v - v_pred

        # Check convergence
        max_error = torch.max(torch.abs(torch.cat([u_error.flatten(), v_error.flatten()])))
        if max_error < tol:
            break

        # Update undistorted coordinates
        u_undist = u_undist + u_error
        v_undist = v_undist + v_error

    return u_undist, v_undist

class Reproject:
    def __init__(self, source_hdu: PrimaryHDU, target_hdu: PrimaryHDU):
        """
        Initialize a reprojection operation between source and target image frames.

        This constructor sets up the necessary components for reprojecting an astronomical
        image from one World Coordinate System (WCS) to another. It stores the source
        and target WCS information, images, and creates a coordinate grid for the target
        image that will be used in the reprojection process.

        Parameters
        ----------
        source_hdu : PrimaryHDU
            HDU containing the data and the header information for the source image

        target_hdu : PrimaryHDU
            HDU containing the data and the header information for the target image


        Notes
        -----
        This constructor creates a coordinate grid spanning the entire target image,
        which will be used for the pixel-to-world and world-to-pixel transformations
        during reprojection. The grid is created with 'ij' indexing, where the first
        dimension corresponds to y (rows) and the second to x (columns).

        The coordinate grid is stored as a tuple of tensors (y_grid, x_grid), where
        each element has the same shape as the target image.
        Examples
        --------
        >>> # Initialize the reprojection object
        >>> reproject = Reproject(target_image, source_image, target_header, source_header, 'cuda:0')

        """
        # Set device
        self.device = get_device()

        # Initialize data
        self.target_image = torch.tensor(target_hdu.data, dtype=torch.float64, device=self.device)
        self.source_image = torch.tensor(source_hdu.data, dtype=torch.float64, device=self.device)

        # Initialize the WCS objects
        self.target_wcs_astropy = WCS(target_hdu.header)
        self.source_wcs_astropy = WCS(source_hdu.header)

        # Define target grid
        self.target_grid = torch.meshgrid(
            torch.arange(self.target_image.shape[0], dtype=torch.float64, device=self.device),  # height
            torch.arange(self.target_image.shape[1], dtype=torch.float64, device=self.device),  # width
            indexing='ij',  # y, x
        )



    def calculate_skyCoords(self, x=None, y=None):
        """Calculate sky coordinates using Astropy WCS implementation."""
        # Get target grid if not provided
        if x is None or y is None:
            y, x  = self.target_grid

        CRPIX1 = self.target_wcs_astropy.wcs.crpix[0]
        CRPIX2 = self.target_wcs_astropy.wcs.crpix[1]
        CRVAL1 = self.target_wcs_astropy.wcs.crval[0]  # Reference RA
        CRVAL2 = self.target_wcs_astropy.wcs.crval[1]  # Reference Dec
        PC_matrix = self.target_wcs_astropy.wcs.get_pc()  # PC Matrix
        CDELT = np.array(self.target_wcs_astropy.wcs.cdelt)  # Scaling factors
        # Get SIP coefficients if present
        sip_coeffs = get_sip_coeffs(self.target_wcs_astropy)
        # Convert numpy arrays to torch tensors if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)
            y = torch.tensor(y, device=self.device)

        # Step 1: Compute Pixel Offsets - Precisely as in wcsprm::p2x
        u = x - (CRPIX1 - 1)
        v = y - (CRPIX2 - 1)
        if sip_coeffs is not None:
            u, v = apply_sip_distortion(u, v, sip_coeffs, self.device)
        # Step 2: Apply PC Matrix (Rotation) and CDELT (Scaling)
        CD_matrix = PC_matrix * CDELT  # Construct CD Matrix
        CD_matrix = torch.tensor(CD_matrix, device=self.device)
        # Handle both scalar and array inputs
        if u.dim() == 0:  # scalar
            pixel_offsets = torch.tensor([u.item(), v.item()], device=self.device)
            transformed = torch.matmul(CD_matrix, pixel_offsets)
            x_scaled, y_scaled = transformed.unbind()
        else:  # arrays
            # Reshape for batch matrix multiplication if needed
            if u.dim() > 1:
                original_shape = u.shape
                u_flat = u.reshape(-1)
                v_flat = v.reshape(-1)
            else:
                u_flat = u
                v_flat = v

            # Stack coordinates for batch processing
            pixel_offsets = torch.stack([u_flat, v_flat], dim=1)  # Shape: [N, 2]

            # Perform batch matrix multiplication
            transformed = torch.matmul(pixel_offsets, CD_matrix.T)  # Shape: [N, 2]
            x_scaled = transformed[:, 0]
            y_scaled = transformed[:, 1]

            # Reshape back to original if needed
            if u.dim() > 1:
                x_scaled = x_scaled.reshape(original_shape)
                y_scaled = y_scaled.reshape(original_shape)

        # Step 3: Use the exact tanx2s logic from WCSLib
        # Compute the radial distance
        r = torch.sqrt(x_scaled ** 2 + y_scaled ** 2)
        r0 = torch.tensor(180.0 / torch.pi, device=self.device)  # R2D from WCSLib

        # Apply the tanx2s function exactly as in WCSLib
        # Note the sign conventions
        phi = torch.zeros_like(r)
        non_zero_r = r != 0
        if torch.any(non_zero_r):
            phi[non_zero_r] = torch.rad2deg(torch.atan2(-x_scaled[non_zero_r], y_scaled[non_zero_r]))

        theta = torch.rad2deg(torch.atan2(r0, r))

        # Step 4: Now apply the sph2x (spherical to native) transform from prjx2s
        # First convert to radians exactly as WCSLib would
        phi_rad = torch.deg2rad(phi)
        theta_rad = torch.deg2rad(theta)
        ra0_rad = torch.tensor(CRVAL1 * torch.pi / 180.0, device=self.device)
        dec0_rad = torch.tensor(CRVAL2 * torch.pi / 180.0, device=self.device)

        # For TAN projection, the pole is at (0,90) in native coordinates
        sin_theta = torch.sin(theta_rad)
        cos_theta = torch.cos(theta_rad)
        sin_phi = torch.sin(phi_rad)
        cos_phi = torch.cos(phi_rad)
        sin_dec0 = torch.sin(dec0_rad)
        cos_dec0 = torch.cos(dec0_rad)

        # This is the exact calculation from wcslib's sphx2s function
        sin_dec = sin_theta * sin_dec0 + cos_theta * cos_dec0 * cos_phi
        dec_rad = torch.arcsin(sin_dec)

        # Calculate RA offset - exact formula from WCSLib
        y_term = cos_theta * sin_phi
        x_term = sin_theta * cos_dec0 - cos_theta * sin_dec0 * cos_phi
        ra_rad = ra0_rad + torch.atan2(-y_term, x_term)

        # Convert to degrees and normalize
        ra = torch.rad2deg(ra_rad) % 360.0
        dec = torch.rad2deg(dec_rad)

        return ra, dec

    def calculate_sourceCoords(self):
        """Calculate source image pixel coordinates corresponding to each target image pixel."""
        # Get sky coordinates in radians
        ra, dec= self.calculate_skyCoords()

        # Get WCS parameters
        CRPIX1 = self.source_wcs_astropy.wcs.crpix[0]
        CRPIX2 = self.source_wcs_astropy.wcs.crpix[1]
        CRVAL1 = self.source_wcs_astropy.wcs.crval[0]  # Reference RA
        CRVAL2 = self.source_wcs_astropy.wcs.crval[1]  # Reference Dec
        PC_matrix = torch.tensor(self.source_wcs_astropy.wcs.get_pc(), device=self.device)
        CDELT = torch.tensor(self.source_wcs_astropy.wcs.cdelt, device=self.device)
        # Get SIP coefficients if present
        sip_coeffs = get_sip_coeffs(self.source_wcs_astropy)
        # Convert numpy arrays to torch tensors if needed
        if not isinstance(ra, torch.Tensor):
            ra = torch.tensor(ra, device=self.device)
            dec = torch.tensor(dec, device=self.device)

        # Helper functions for trigonometric calculations
        def atan2d(y, x):
            """PyTorch implementation of WCSLib's atan2d function"""
            return torch.rad2deg(torch.atan2(y, x))

        def sincosd(angle_deg):
            """PyTorch implementation of WCSLib's sincosd function"""
            angle_rad = torch.deg2rad(angle_deg)
            return torch.sin(angle_rad), torch.cos(angle_rad)

        # Step 1: Convert from world to native spherical coordinates
        # Convert to radians
        ra_rad = torch.deg2rad(ra)
        dec_rad = torch.deg2rad(dec)
        ra0_rad = torch.tensor(CRVAL1 * torch.pi / 180.0, device=self.device)
        dec0_rad = torch.tensor(CRVAL2 * torch.pi / 180.0, device=self.device)

        # Calculate the difference in RA
        delta_ra = ra_rad - ra0_rad

        # Calculate sine and cosine values
        sin_dec = torch.sin(dec_rad)
        cos_dec = torch.cos(dec_rad)
        sin_dec0 = torch.sin(dec0_rad)
        cos_dec0 = torch.cos(dec0_rad)
        sin_delta_ra = torch.sin(delta_ra)
        cos_delta_ra = torch.cos(delta_ra)

        # Calculate the native spherical coordinates using the correct sign conventions
        # Calculate the numerator for phi (native longitude)
        y_phi = -cos_dec * sin_delta_ra  # Note the negative sign

        # Calculate the denominator for phi
        x_phi = sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_delta_ra

        # Calculate native longitude (phi)
        phi = atan2d(y_phi, x_phi)

        # Calculate native latitude (theta)
        theta = torch.rad2deg(torch.arcsin(sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_delta_ra))

        # Step 2: Apply the TAN projection (tans2x function from WCSLib)
        # Calculate sine and cosine of phi and theta
        sin_phi, cos_phi = sincosd(phi)
        sin_theta, cos_theta = sincosd(theta)

        # Check for singularity (when sin_theta is zero)
        eps = 1e-10
        if torch.any(torch.abs(sin_theta) < eps):
            raise ValueError("Singularity in tans2x: theta close to 0 degrees")

        # r0 is the radius scaling factor (typically 180.0/π)
        r0 = torch.tensor(180.0 / torch.pi, device=self.device)

        # Calculate the scaling factor r with correct sign
        r = r0 * cos_theta / sin_theta

        # Calculate intermediate world coordinates (x_scaled, y_scaled)
        # With the corrected signs based on your findings
        x_scaled = -r * sin_phi  # Note the negative sign
        y_scaled = r * cos_phi

        # Step 3: Apply the inverse of the CD matrix to get pixel offsets
        # First, construct the CD matrix
        CD_matrix = PC_matrix * CDELT
        CD_matrix = torch.tensor(CD_matrix, device=self.device)
        # Calculate the inverse of the CD matrix
        CD_inv = torch.linalg.inv(CD_matrix)

        # Handle batch processing for arrays
        if ra.dim() == 0:  # scalar inputs
            standard_coords = torch.tensor([x_scaled.item(), y_scaled.item()], device=self.device)
            pixel_offsets = torch.matmul(CD_inv, standard_coords)
            u = pixel_offsets[0]
            v = pixel_offsets[1]
        else:  # array inputs
            # Reshape for batch processing if needed
            if ra.dim() > 1:
                original_shape = ra.shape
                x_scaled_flat = x_scaled.reshape(-1)
                y_scaled_flat = y_scaled.reshape(-1)
            else:
                x_scaled_flat = x_scaled
                y_scaled_flat = y_scaled

            # Stack for batch matrix multiplication
            standard_coords = torch.stack([x_scaled_flat, y_scaled_flat], dim=1)  # Shape: [N, 2]

            # Use batch matrix multiplication
            pixel_offsets = torch.matmul(standard_coords, CD_inv.T)  # Shape: [N, 2]
            u = pixel_offsets[:, 0]
            v = pixel_offsets[:, 1]

            # Reshape back to original dimensions if needed
            if ra.dim() > 1:
                u = u.reshape(original_shape)
                v = v.reshape(original_shape)

        if sip_coeffs is not None:
            u, v = apply_inverse_sip_distortion(u, v, sip_coeffs, self.device)

        # Step 4: Add the reference pixel to get final pixel coordinates
        # Remember to add (CRPIX-1) to account for 1-based indexing in FITS/WCS
        x_pixel = u + (CRPIX1 - 1)
        y_pixel = v + (CRPIX2 - 1)

        return x_pixel, y_pixel

    def interpolate_source_image(self, interpolation_mode='bilinear'):
        """
        Interpolate the source image at the calculated source coordinates with flux conservation.

        This method performs the actual pixel resampling needed for reprojection
        while preserving the total flux (photometric accuracy). It implements a
        footprint-based approach similar to that used in reproject_interp from the
        Astropy package.

        The method uses a combined tensor approach for computational efficiency,
        performing both image resampling and footprint tracking in a single operation.
        Total flux is conserved both locally (via footprint correction) and globally
        (via final normalization).

        Parameters
        ----------
        interpolation_mode : str, default 'bilinear'
            The interpolation mode to use when sampling the source image.
            Options include:
            - 'nearest' : Nearest neighbor interpolation (no interpolation)
            - 'bilinear' : Bilinear interpolation (default)
            - 'bicubic' : Bicubic interpolation

            These correspond to the modes available in torch.nn.functional.grid_sample.

        Returns
        -------
        torch.Tensor
            The reprojected image with the same shape as the target image.
            Pixel values are interpolated from the source image according to
            the WCS transformation with flux conservation preserved.

        Notes
        -----
        This implementation uses a two-step flux conservation approach:

        1. Local flux conservation: The image and a "ones" tensor are interpolated
           together, and the interpolated image is divided by the interpolated ones
           tensor (footprint) to correct for any flux spreading during interpolation.

        2. Global flux conservation: The total flux of the output image is normalized
           to match the total flux of the input image.

        Areas in the target image that map outside the source image boundaries
        will be filled with zeros (using 'zeros' padding_mode).

        This method is particularly suitable for high-precision photometry with
        extended sources, as it properly preserves both the background noise
        characteristics and the flux distribution of sources.
        """
        # Get source coordinates
        x_source, y_source = self.calculate_sourceCoords()
        H, W = self.source_image.shape

        # Original grid_sample implementation for other modes
        x_normalized = 2.0 * (x_source / (W - 1)) - 1.0
        y_normalized = 2.0 * (y_source / (H - 1)) - 1.0

        # Calculate origin flux
        #original_total_flux = torch.sum(self.source_image)
        # Stack coordinates into sampling grid
        grid = torch.stack([x_normalized, y_normalized], dim=-1)

        # Add batch and channel dimensions if needed
        source_image = self.source_image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        grid = grid.unsqueeze(0)  # [1, H, W, 2]
        ones = torch.ones_like(source_image)
        # Stack image and ones along the channel dimension
        combined = torch.cat([source_image, ones], dim=1)  # Shape becomes [1, 2, H, W]

        # Single grid_sample call
        combined_result = interpolate_image(combined, grid, interpolation_mode)

        # Split the results
        resampled = combined_result[:, 0].squeeze()
        footprint = combined_result[:, 1].squeeze()
        # Apply footprint correction where the footprint is significant
        valid_mask = footprint > 1e-6
        resampled[valid_mask] /= footprint[valid_mask]
        # Apply simple flux conservation
        #new_total_flux = torch.sum(resampled)
        #normalization_factor_flux = original_total_flux / new_total_flux if new_total_flux > 0 else 1

        return resampled #* normalization_factor_flux


def calculate_reprojection(source_hdu: fits.PrimaryHDU, target_hdu: fits.PrimaryHDU, interpolation_mode='nearest'):
    """
    Reproject an astronomical image from a source WCS to a target WCS.

    This high-level function provides a convenient interface for image reprojection,
    handling all the necessary steps: WCS extraction, tensor creation, and interpolation.
    It converts FITS HDU objects to the internal representation, performs the reprojection,
    and returns the resulting image as a PyTorch tensor.

    Parameters
    ----------
    source_hdu : fits.PrimaryHDU
        The source image HDU containing the image data to be reprojected and
        its associated WCS information in the header.

    target_hdu : fits.PrimaryHDU
        The target image HDU providing the output grid and WCS information. The
        shape of target_hdu.data defines the dimensions of the output image.

    interpolation_mode : str, default 'nearest'
        The interpolation method to use when resampling the source image.
        Options:
        - 'nearest' : Nearest neighbor interpolation (fastest, default)
        - 'bilinear' : Bilinear interpolation (good balance of speed/quality)
        - 'bicubic' : Bicubic interpolation (highest quality, slowest)

    Returns
    -------
    torch.Tensor
        The reprojected image as a PyTorch tensor with the same shape as
        target_hdu.data. The tensor is on the same device as the computation
        (GPU if available, otherwise CPU).

    Notes
    -----
    This function automatically:
    - Detects and uses GPU acceleration if available
    - Handles byte order conversion for tensor creation
    - Converts data to float32 for processing
    - Creates WCSHeader objects from FITS headers

    To save the result as a FITS file, you will need to convert the tensor
    back to a NumPy array and create a new FITS HDU with the target WCS header.

    Examples
    --------
    >>> from astropy.io import fits
    >>> import torch
    >>> from reprojection.reproject import calculate_reprojection
    >>>
    >>> # Open source and target images
    >>> source_hdu = fits.open('source_image.fits')[0]
    >>> target_hdu = fits.open('target_grid.fits')[0]
    >>>
    >>> # Perform reprojection with bilinear interpolation
    >>> reprojected = calculate_reprojection(
    ...     target_hdu=target_hdu,
    ...     source_hdu=source_hdu,
    ...     interpolation_mode='bilinear'
    ... )
    >>>
    >>> # Convert back to NumPy and save as FITS
    >>> reprojected_np = reprojected.cpu().numpy()
    >>> output_hdu = fits.PrimaryHDU(data=reprojected_np, header=target_hdu.header)
    >>> output_hdu.writeto('reprojected_image.fits', overwrite=True)
    """
    reprojection = Reproject(source_hdu=source_hdu, target_hdu=target_hdu)
    return reprojection.interpolate_source_image(interpolation_mode=interpolation_mode)