import numpy as np
import torch
from astropy.io import fits
from dataclasses import dataclass
from typing import List
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


@dataclass
class WCSHeader:
    """
    A dataclass containing World Coordinate System (WCS) parameters with SIP distortion coefficients.

    This class encapsulates all the necessary parameters for defining a WCS transformation
    with Simple Imaging Polynomial (SIP) distortion representation. It supports conversion
    between pixel coordinates and world coordinates (RA/Dec) while accounting for optical
    distortions in astronomical images.


    """
    CRVAL1: torch.tensor
    CRVAL2: torch.tensor
    CRPIX1: torch.tensor
    CRPIX2: torch.tensor
    CDELT1: torch.tensor
    CDELT2: torch.tensor
    A_ORDER: int
    B_ORDER: int
    AP_ORDER: int
    BP_ORDER: int
    A: List[float]
    B: List[float]
    AP: List[float]
    BP: List[float]
    PC_Matrix: torch.tensor

    @classmethod
    def from_header(cls, header: fits.Header, device='cpu'):
        """
        Create a WCSHeader instance from an Astropy FITS header.

        This class method extracts World Coordinate System (WCS) information from a FITS header,
        including SIP distortion coefficients if present, and converts them to PyTorch tensors
        for use in the WCSHeader class.

        Parameters
        ----------
        header : astropy.io.fits.Header
            An Astropy FITS header object containing WCS information. At minimum, the header
            should contain the basic WCS keywords (CRVAL1/2, CRPIX1/2, CDELT1/2). If SIP
            distortion is present, it will extract the polynomial coefficients.

        Returns
        -------
        WCSHeader
            A new instance of WCSHeader initialized with the WCS information from the header.

        Notes
        -----
        This method:
        - Extracts basic WCS parameters (reference coordinates, pixel scales)
        - Handles SIP distortion polynomials (A, B, AP, BP coefficients)
        - Sets default values for missing parameters
        - Converts all numeric values to PyTorch tensors with float32 precision

        The method is tolerant of missing SIP parameters and will default to an identity
        transformation if SIP coefficients are not present in the header.

        The PC matrix (rotation/scaling matrix) defaults to the identity matrix if not specified.

        Examples
        --------
        >>> from astropy.io import fits
        >>> from reprojection import WCSHeader
        >>>
        >>> # Open a FITS file and read its header
        >>> hdul = fits.open('image.fits')
        >>> header = hdul[0].header
        >>>
        >>> # Create a WCSHeader instance from the FITS header
        >>> wcs_header = WCSHeader.from_header(header)
        >>>
        >>> # Now you can use the wcs_header for coordinate transformations
        >>> # or pass it to reprojection functions

        Raises
        ------
        KeyError
            If the header is missing required WCS keywords (CRVAL1/2, CRPIX1/2, CDELT1/2)
        """
        # Get reference coordinates and convert to torch tensors
        wcs_info = {'CRVAL2': torch.tensor(header['CRVAL2'], dtype=torch.float64),
                    'CRVAL1': torch.tensor(header['CRVAL1'], dtype=torch.float64),
                    'CRPIX1': torch.tensor(header['CRPIX1'], dtype=torch.float64),
                    'CRPIX2': torch.tensor(header['CRPIX2'], dtype=torch.float64),
                    'CDELT1': torch.tensor(header['CDELT1'], dtype=torch.float64),
                    'CDELT2': torch.tensor(header['CDELT2'], dtype=torch.float64),
                    'CTYPE1': torch.tensor(header['CTYPE1'], dtype=torch.float64),
                    'CTYPE2': torch.tensor(header['CTYPE2'], dtype=torch.float64),
                    'A_ORDER': header.get('A_ORDER', 0),
                    'B_ORDER': header.get('B_ORDER', 0), 'AP_ORDER': header.get('AP_ORDER', 0),
                    'BP_ORDER': header.get('BP_ORDER', 0)}

        # Get SIP orders

        # Get PC Matrix values
        pc_matrix = torch.tensor([
            [header.get('PC1_1', 1.0), header.get('PC1_2', 0.0)],
            [header.get('PC2_1', 0.0), header.get('PC2_2', 1.0)]
        ], dtype=torch.float64, device=device)
        wcs_info['PC_Matrix'] = pc_matrix
        cdelt = torch.tensor([wcs_info['CDELT1'], wcs_info['CDELT2']], dtype=torch.float64, device=device).diag()
        wcs_info['CDELT'] = cdelt
        cd = torch.matmul(cdelt, pc_matrix)
        wcs_info['cd'] = cd
        # Calculate and store the inverse CD matrix for world to pixel transformations
        det = cd[0, 0] * cd[1, 1] - cd[0, 1] * cd[1, 0]
        cd_inv = torch.zeros_like(cd)
        cd_inv[0, 0] = cd[1, 1] / det
        cd_inv[0, 1] = -cd[0, 1] / det
        cd_inv[1, 0] = -cd[1, 0] / det
        cd_inv[1, 1] = cd[0, 0] / det
        wcs_info['cd_inv'] = cd_inv
        # Get projection type TODO: UPDATE TO ACCEPT OTHER THAN TAN
        projection = 'TAN'
        wcs_info['projection'] = projection
        # Initialize empty coefficient lists
        sip_a = []
        sip_b = []
        sip_ap = []
        sip_bp = []

        # Read SIP coefficients if they exist
        if wcs_info['A_ORDER'] > 0:
            sip_a = [
                header.get('A_0_2', 0.0), header.get('A_0_3', 0.0),
                header.get('A_1_1', 0.0), header.get('A_1_2', 0.0),
                header.get('A_2_0', 0.0), header.get('A_2_1', 0.0),
                header.get('A_3_0', 0.0)
            ]
            sip_b = [
                header.get('B_0_2', 0.0), header.get('B_0_3', 0.0),
                header.get('B_1_1', 0.0), header.get('B_1_2', 0.0),
                header.get('B_2_0', 0.0), header.get('B_2_1', 0.0),
                header.get('B_3_0', 0.0)
            ]

        # Read inverse SIP coefficients if they exist
        if wcs_info['AP_ORDER'] > 0:
            sip_ap = [
                header.get('AP_0_0', 0.0), header.get('AP_0_1', 0.0),
                header.get('AP_0_2', 0.0), header.get('AP_0_3', 0.0),
                header.get('AP_1_0', 0.0), header.get('AP_1_1', 0.0),
                header.get('AP_1_2', 0.0), header.get('AP_2_0', 0.0),
                header.get('AP_2_1', 0.0), header.get('AP_3_0', 0.0)
            ]
            sip_bp = [
                header.get('BP_0_0', 0.0), header.get('BP_0_1', 0.0),
                header.get('BP_0_2', 0.0), header.get('BP_0_3', 0.0),
                header.get('BP_1_0', 0.0), header.get('BP_1_1', 0.0),
                header.get('BP_1_2', 0.0), header.get('BP_2_0', 0.0),
                header.get('BP_2_1', 0.0), header.get('BP_3_0', 0.0)
            ]

        wcs_info['sip_a'] = sip_a
        wcs_info['sip_b'] = sip_b
        wcs_info['sip_ap'] = sip_ap
        wcs_info['sip_ab'] = sip_bp
        wcs_info['sip_crpix1'] = wcs_info['CRPIX1']
        wcs_info['sip_crpix2'] = wcs_info['CRPIX2']

        return cls(**wcs_info)



class Reproject:
    def __init__(self, target_wcs: WCSHeader, source_wcs: WCSHeader, target_image: torch.Tensor,
                 source_image: torch.Tensor, target_header, source_header,
                 device: str):
        """
        Initialize a reprojection operation between source and target image frames.

        This constructor sets up the necessary components for reprojecting an astronomical
        image from one World Coordinate System (WCS) to another. It stores the source
        and target WCS information, images, and creates a coordinate grid for the target
        image that will be used in the reprojection process.

        Parameters
        ----------
        target_wcs : WCSHeader
            The WCS information for the target image, containing the coordinate system
            and distortion parameters that define the output frame.

        source_wcs : WCSHeader
            The WCS information for the source image, containing the coordinate system
            and distortion parameters of the input frame.

        target_image : torch.Tensor
            The target image tensor, which defines the shape and resolution of the
            output reprojected image. This may be an empty tensor with the desired dimensions.

        source_image : torch.Tensor
            The source image tensor containing the pixel data to be reprojected.

        device : str
            The device to perform computations on, either 'cpu' or a CUDA device
            specification (e.g., 'cuda:0'). All tensors will be created on this device.

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
        >>> # Assuming you have WCSHeader objects and image tensors:
        >>> target_wcs = WCSHeader.from_header(target_fits_header)
        >>> source_wcs = WCSHeader.from_header(source_fits_header)
        >>>
        >>> # Create tensor versions of images
        >>> target_img = torch.zeros((1024, 1024), dtype=torch.float32)
        >>> source_img = torch.tensor(source_data, dtype=torch.float32)
        >>>
        >>> # Initialize the reprojection object
        >>> reproject = Reproject(target_wcs, source_wcs, target_img, source_img, 'cuda:0')

        """
        self.target_wcs = target_wcs
        self.source_wcs = source_wcs
        self.target_image = target_image
        self.source_image = source_image

        self.target_header = target_header
        self.source_header = source_header
        self.device = device

        # Initialize the TorchWCS objects
        self.target_wcs_astropy = WCS(target_header)
        self.source_wcs_astropy = WCS(source_header)

        self.target_grid = torch.meshgrid(
            torch.arange(self.target_image.shape[0], dtype=torch.float64, device=device),  # height
            torch.arange(self.target_image.shape[1], dtype=torch.float64, device=device),  # width
            indexing='ij',  # y, x
        )

    def calculate_skyCoords(self, x=None, y=None):
        """Calculate sky coordinates using Astropy WCS implementation."""
        # Get target grid if not provided
        if x is None or y is None:
            y, x = self.target_grid

        # Flatten x and y arrays
        height, width = x.shape
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)

        # Use astropy's pixel_to_world_values to convert pixel coordinates to world coordinates
        ra_deg, dec_deg = self.target_wcs_astropy.pixel_to_world_values(x_flat, y_flat)

        # Reshape to original grid shape
        ra_deg = ra_deg.reshape(height, width)
        dec_deg = dec_deg.reshape(height, width)

        # Convert to radians
        ra_rad = ra_deg * (torch.pi / 180.0)
        dec_rad = dec_deg * (torch.pi / 180.0)

        return ra_rad, dec_rad

    def calculate_sourceCoords(self):
        """Calculate source image pixel coordinates corresponding to each target image pixel."""
        # Get sky coordinates in radians
        ra_rad, dec_rad = self.calculate_skyCoords()

        # Convert to degrees for the WCS transformation
        ra_deg = ra_rad * (180.0 / torch.pi)
        dec_deg = dec_rad * (180.0 / torch.pi)

        # Flatten arrays
        height, width = ra_deg.shape
        ra_flat = ra_deg.reshape(-1)
        dec_flat = dec_deg.reshape(-1)

        # Use astropy's world_to_pixel_values to convert world coordinates to source pixel coordinates
        x_source_flat, y_source_flat = self.source_wcs_astropy.world_to_pixel_values(ra_flat, dec_flat)

        # Convert back to torch tensors
        x_source_flat = torch.tensor(x_source_flat, dtype=torch.float64, device=ra_rad.device)
        y_source_flat = torch.tensor(y_source_flat, dtype=torch.float64, device=ra_rad.device)

        # Reshape back to original dimensions
        x_source = x_source_flat.reshape(height, width)
        y_source = y_source_flat.reshape(height, width)

        return x_source, y_source

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
        original_total_flux = torch.sum(self.source_image)
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
        new_total_flux = torch.sum(resampled)
        normalization_factor_flux = original_total_flux / new_total_flux if new_total_flux > 0 else 1

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
    >>> from reprojection import calculate_reprojection
    >>>
    >>> # Open source and target images
    >>> source_hdu = fits.open('source_image.fits')[0]
    >>> target_hdu = fits.open('target_grid.fits')[0]
    >>>
    >>> # Perform reprojection with bilinear interpolation
    >>> reprojected = calculate_reprojection(
    ...     source_hdu=source_hdu,
    ...     target_hdu=target_hdu,
    ...     interpolation_mode='bilinear'
    ... )
    >>>
    >>> # Convert back to NumPy and save as FITS
    >>> reprojected_np = reprojected.cpu().numpy()
    >>> output_hdu = fits.PrimaryHDU(data=reprojected_np, header=target_hdu.header)
    >>> output_hdu.writeto('reprojected_image.fits', overwrite=True)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #target_wcs = WCSHeader.from_header(target_hdu.header, device=device)
    #source_wcs = WCSHeader.from_header(source_hdu.header, device=device)
    # Convert the data to native byte order before creating tensors
    target_data = target_hdu.data.astype(target_hdu.data.dtype.newbyteorder('='))
    source_data = source_hdu.data.astype(source_hdu.data.dtype.newbyteorder('='))
    # Now create the tensors
    target_image = torch.tensor(target_data, dtype=torch.float64, device=device)
    source_image = torch.tensor(source_data, dtype=torch.float64, device=device)
    reprojection = Reproject(target_wcs=target_hdu, source_wcs=source_hdu, target_image=target_image,
                             source_image=source_image, target_header=target_hdu.header, source_header=source_hdu.header,
                             device=device)
    return reprojection.interpolate_source_image(interpolation_mode=interpolation_mode)