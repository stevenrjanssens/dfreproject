import torch
from astropy.io import fits
from dataclasses import dataclass
from typing import List

EPSILON = 1e-10


@dataclass
class WCSHeader:
    """
    Dataclass to hold all WCS information
    """
    DEC_0: torch.tensor  # Declination in degrees
    RA_0: torch.tensor   # RA in degrees
    CRPIX1: torch.tensor
    CRPIX2: torch.tensor
    CDELT1: torch.tensor
    CDELT2: torch.tensor
    A_ORDER: int  # Order of SIP polynomial A
    B_ORDER: int  # Order of SIP polynomial B
    AP_ORDER: int  # Order of inverse SIP polynomial A
    BP_ORDER: int  # Order of inverse SIP polynomial B
    A: List[float]  # SIP Polynomial A coefficients: [A_0_2, A_0_3, A_1_1, A_1_2, A_2_0, A_2_1, A_3_0]
    B: List[float]  # SIP Polynomial B coefficients: [B_0_2, B_0_3, B_1_1, B_1_2, B_2_0, B_2_1, B_3_0]
    AP: List[float]  # Inverse SIP Polynomial A coefficients: [AP_0_0, AP_0_1, AP_0_2, AP_0_3, AP_1_0, AP_1_1, AP_1_2, AP_2_0, AP_2_1, AP_3_0]
    BP: List[float]  # Inverse SIP Polynomial B coefficients: [BP_0_0, BP_0_1, BP_0_2, BP_0_3, BP_1_0, BP_1_1, BP_1_2, BP_2_0, BP_2_1, BP_3_0]
    PC_Matrix: torch.tensor  # Matrix of PC values as torch.Tensor: [[PC1_1, PC1_2], [PC2_1, PC2_2]]

    @classmethod
    def from_header(cls, header):
        """
        Create WCSHeader from astropy FITS header

        Args:
            header: astropy.io.fits header object
        Returns:
            WCSHeader: Instance of WCSHeader containing WCS information
        """
        # Get reference coordinates and convert to torch tensors
        wcs_info = {
            'DEC_0': torch.tensor(header['CRVAL2'], dtype=torch.float32),
            'RA_0': torch.tensor(header['CRVAL1'], dtype=torch.float32),
            'CRPIX1': torch.tensor(header['CRPIX1'], dtype=torch.float32),
            'CRPIX2': torch.tensor(header['CRPIX2'], dtype=torch.float32),
            'CDELT1': torch.tensor(header['CDELT1'], dtype=torch.float32),
            'CDELT2': torch.tensor(header['CDELT2'], dtype=torch.float32),
        }

        # Get SIP orders
        wcs_info['A_ORDER'] = header.get('A_ORDER', 0)
        wcs_info['B_ORDER'] = header.get('B_ORDER', 0)
        wcs_info['AP_ORDER'] = header.get('AP_ORDER', 0)
        wcs_info['BP_ORDER'] = header.get('BP_ORDER', 0)

        # Get PC Matrix values
        pc_matrix = torch.tensor([
            [header.get('PC1_1', 1.0), header.get('PC1_2', 0.0)],
            [header.get('PC2_1', 0.0), header.get('PC2_2', 1.0)]
        ], dtype=torch.float32)
        wcs_info['PC_Matrix'] = pc_matrix
        # Initialize empty coefficient lists
        A_coeffs = []
        B_coeffs = []
        AP_coeffs = []
        BP_coeffs = []

        # Read SIP coefficients if they exist
        if wcs_info['A_ORDER'] > 0:
            A_coeffs = [
                header.get('A_0_2', 0.0), header.get('A_0_3', 0.0),
                header.get('A_1_1', 0.0), header.get('A_1_2', 0.0),
                header.get('A_2_0', 0.0), header.get('A_2_1', 0.0),
                header.get('A_3_0', 0.0)
            ]
            B_coeffs = [
                header.get('B_0_2', 0.0), header.get('B_0_3', 0.0),
                header.get('B_1_1', 0.0), header.get('B_1_2', 0.0),
                header.get('B_2_0', 0.0), header.get('B_2_1', 0.0),
                header.get('B_3_0', 0.0)
            ]

        # Read inverse SIP coefficients if they exist
        if wcs_info['AP_ORDER'] > 0:
            AP_coeffs = [
                header.get('AP_0_0', 0.0), header.get('AP_0_1', 0.0),
                header.get('AP_0_2', 0.0), header.get('AP_0_3', 0.0),
                header.get('AP_1_0', 0.0), header.get('AP_1_1', 0.0),
                header.get('AP_1_2', 0.0), header.get('AP_2_0', 0.0),
                header.get('AP_2_1', 0.0), header.get('AP_3_0', 0.0)
            ]
            BP_coeffs = [
                header.get('BP_0_0', 0.0), header.get('BP_0_1', 0.0),
                header.get('BP_0_2', 0.0), header.get('BP_0_3', 0.0),
                header.get('BP_1_0', 0.0), header.get('BP_1_1', 0.0),
                header.get('BP_1_2', 0.0), header.get('BP_2_0', 0.0),
                header.get('BP_2_1', 0.0), header.get('BP_3_0', 0.0)
            ]

        wcs_info['A'] = A_coeffs
        wcs_info['B'] = B_coeffs
        wcs_info['AP'] = AP_coeffs
        wcs_info['BP'] = BP_coeffs

        return cls(**wcs_info)

    def inverse_PC_Matrix(self):
        """
        Calculate inverse PC matrix
        Returns:
            Inverse PC Matrix
        """
        return torch.linalg.inv(self.PC_Matrix)

    def SIP_polynomial_A(self, u, v):
        """
        Calculate SIP polynomial A

        Args:
            u (torch.Tensor): intermediate x coordinates
            v (torch.Tensor): intermediate y coordinates

        Returns:
            torch.Tensor: value of SIP polynomial A
        """
        if self.A_ORDER == 2:
            # For order 2, we expect different coefficients
            # Assuming list is in this order: [A_0_2, A_1_1, A_2_0]
            if len(self.A) >= 3:
                A_0_2 = self.A[0]
                A_1_1 = self.A[1]
                A_2_0 = self.A[2]

                return (A_0_2 * v ** 2 +
                        A_1_1 * u * v +
                        A_2_0 * u ** 2)
            else:
                return torch.zeros_like(u)  # Safe fallback if not enough coefficients

        elif self.A_ORDER == 3:
            if len(self.A) >= 7:
                A_0_2 = self.A[0]
                A_0_3 = self.A[1]
                A_1_1 = self.A[2]
                A_1_2 = self.A[3]
                A_2_0 = self.A[4]
                A_2_1 = self.A[5]
                A_3_0 = self.A[6]

                return (A_0_2 * v ** 2 + A_0_3 * v ** 3 +
                        A_1_1 * u * v + A_1_2 * u * v ** 2 +
                        A_2_0 * u ** 2 + A_2_1 * u ** 2 * v +
                        A_3_0 * u ** 3)
            else:
                return torch.zeros_like(u)  # Safe fallback

        else:
            return torch.zeros_like(u)  # Return zeros for unsupported orders

    def SIP_polynomial_B(self, u, v):
        """
        Calculate SIP polynomial B

        Args:
            u (torch.Tensor): intermediate x coordinates
            v (torch.Tensor): intermediate y coordinates

        Returns:
            torch.Tensor: value of SIP polynomial B
        """
        if self.B_ORDER == 2:
            # For order 2, we expect different coefficients
            # Assuming list is in this order: [B_0_2, B_1_1, B_2_0]
            if len(self.B) >= 3:
                B_0_2 = self.B[0]
                B_1_1 = self.B[1]
                B_2_0 = self.B[2]

                return (B_0_2 * v ** 2 +
                        B_1_1 * u * v +
                        B_2_0 * u ** 2)
            else:
                return torch.zeros_like(u)  # Safe fallback

        elif self.B_ORDER == 3:
            if len(self.B) >= 7:
                B_0_2 = self.B[0]
                B_0_3 = self.B[1]
                B_1_1 = self.B[2]
                B_1_2 = self.B[3]
                B_2_0 = self.B[4]
                B_2_1 = self.B[5]
                B_3_0 = self.B[6]

                return (B_0_2 * v ** 2 + B_0_3 * v ** 3 +
                        B_1_1 * u * v + B_1_2 * u * v ** 2 +
                        B_2_0 * u ** 2 + B_2_1 * u ** 2 * v +
                        B_3_0 * u ** 3)
            else:
                return torch.zeros_like(u)  # Safe fallback

        else:
            return torch.zeros_like(u)  # Return zeros for unsupported orders

    def SIP_polynomial_AP(self, u, v):
        """
        Compute inverse SIP polynomial A

        Args:
            u (torch.Tensor): intermediate x coordinates
            v (torch.Tensor): intermediate y coordinates

        Returns:
            torch.Tensor: value of inverse SIP polynomial A
        """
        if self.AP_ORDER == 2:
            # For order 2, using the first 6 coefficients:
            # [AP_0_0, AP_0_1, AP_0_2, AP_1_0, AP_1_1, AP_2_0]
            if len(self.AP) >= 6:
                AP_0_0 = self.AP[0]
                AP_0_1 = self.AP[1]
                AP_0_2 = self.AP[2]
                AP_1_0 = self.AP[3]
                AP_1_1 = self.AP[4]
                AP_2_0 = self.AP[5]

                return (AP_0_0 +
                        AP_0_1 * v + AP_0_2 * v ** 2 +
                        AP_1_0 * u + AP_1_1 * u * v +
                        AP_2_0 * u ** 2)
            else:
                return torch.zeros_like(u)  # Safe fallback

        elif self.AP_ORDER == 3:
            if len(self.AP) >= 10:
                AP_0_0 = self.AP[0]
                AP_0_1 = self.AP[1]
                AP_0_2 = self.AP[2]
                AP_0_3 = self.AP[3]
                AP_1_0 = self.AP[4]
                AP_1_1 = self.AP[5]
                AP_1_2 = self.AP[6]
                AP_2_0 = self.AP[7]
                AP_2_1 = self.AP[8]
                AP_3_0 = self.AP[9]

                return (AP_0_0 +
                        AP_0_1 * v + AP_0_2 * v ** 2 + AP_0_3 * v ** 3 +
                        AP_1_0 * u + AP_1_1 * u * v + AP_1_2 * u * v ** 2 +
                        AP_2_0 * u ** 2 + AP_2_1 * u ** 2 * v +
                        AP_3_0 * u ** 3)
            else:
                return torch.zeros_like(u)  # Safe fallback

        else:
            return torch.zeros_like(u)  # Return zeros for unsupported orders

    def SIP_polynomial_BP(self, u, v):
        """
        Compute inverse SIP polynomial B

        Args:
            u (torch.Tensor): intermediate x coordinates
            v (torch.Tensor): intermediate y coordinates

        Returns:
            torch.Tensor: value of inverse SIP polynomial B
        """
        if self.BP_ORDER == 2:
            # For order 2, using the first 6 coefficients:
            # [BP_0_0, BP_0_1, BP_0_2, BP_1_0, BP_1_1, BP_2_0]
            if len(self.BP) >= 6:
                BP_0_0 = self.BP[0]
                BP_0_1 = self.BP[1]
                BP_0_2 = self.BP[2]
                BP_1_0 = self.BP[3]
                BP_1_1 = self.BP[4]
                BP_2_0 = self.BP[5]

                return (BP_0_0 +
                        BP_0_1 * v + BP_0_2 * v ** 2 +
                        BP_1_0 * u + BP_1_1 * u * v +
                        BP_2_0 * u ** 2)
            else:
                return torch.zeros_like(u)  # Safe fallback

        elif self.BP_ORDER == 3:
            if len(self.BP) >= 10:
                BP_0_0 = self.BP[0]
                BP_0_1 = self.BP[1]
                BP_0_2 = self.BP[2]
                BP_0_3 = self.BP[3]
                BP_1_0 = self.BP[4]
                BP_1_1 = self.BP[5]
                BP_1_2 = self.BP[6]
                BP_2_0 = self.BP[7]
                BP_2_1 = self.BP[8]
                BP_3_0 = self.BP[9]

                return (BP_0_0 +
                        BP_0_1 * v + BP_0_2 * v ** 2 + BP_0_3 * v ** 3 +
                        BP_1_0 * u + BP_1_1 * u * v + BP_1_2 * u * v ** 2 +
                        BP_2_0 * u ** 2 + BP_2_1 * u ** 2 * v +
                        BP_3_0 * u ** 3)
            else:
                return torch.zeros_like(u)  # Safe fallback

        else:
            return torch.zeros_like(u)  # Return zeros for unsupported orders



class Reproject:
    def __init__(self, target_wcs: WCSHeader, source_wcs: WCSHeader, target_image: torch.Tensor, source_image: torch.Tensor, device: str):
        self.target_wcs = target_wcs
        self.source_wcs = source_wcs
        self.target_image = target_image
        self.source_image = source_image

        self.target_grid = torch.meshgrid(
            torch.arange(self.target_image.shape[0], dtype=torch.float32, device=device),  # height
            torch.arange(self.target_image.shape[1], dtype=torch.float32, device=device),  # width
            indexing='ij',  # y, x
        )

    def calculate_skyCoords(self):
        """
        Calculate sky coordinates from target image coordinates.
        """
        y, x = self.target_grid  # get target grid

        # Convert to offset from reference pixel
        u = x - self.target_wcs.CRPIX1
        v = y - self.target_wcs.CRPIX2

        # Apply SIP distortion correction
        x_distorted = u + self.target_wcs.SIP_polynomial_A(u, v)
        y_distorted = v + self.target_wcs.SIP_polynomial_B(u, v)

        # Combine distorted coordinates
        coords = torch.stack([x_distorted, y_distorted])
        coords_flat = coords.reshape(2, -1)

        # Apply PC matrix transformation
        transformed = torch.einsum('ij,jk->ik', self.target_wcs.PC_Matrix, coords_flat)

        # Reshape back to image dimensions
        x_intermediate = transformed[0].reshape(self.target_image.shape[0], self.target_image.shape[1])
        y_intermediate = transformed[1].reshape(self.target_image.shape[0], self.target_image.shape[1])

        # Scale by pixel scale
        x_scaled = x_intermediate * self.target_wcs.CDELT1
        y_scaled = y_intermediate * self.target_wcs.CDELT2

        # x_scaled = x_scaled * torch.pi / 180
        # y_scaled = y_scaled * torch.pi / 180


        # Convert reference coordinates to radians
        dec_0_rad = torch.deg2rad(torch.tensor(self.target_wcs.DEC_0, dtype=x_scaled.dtype))
        ra_0_rad = torch.deg2rad(torch.tensor(self.target_wcs.RA_0, dtype=x_scaled.dtype))

        # Compute radial distance
        r = torch.hypot(x_scaled, y_scaled)

        # Compute angular distance from reference point
        theta = torch.atan(r)

        # Compute position angle
        phi = torch.atan2(-x_scaled, y_scaled)

        # Compute Declination
        dec = torch.asin(
            torch.sin(dec_0_rad) * torch.cos(theta) +
            torch.cos(dec_0_rad) * torch.sin(theta) * torch.cos(phi)
        )

        # Compute Right Ascension
        ra = ra_0_rad + torch.atan2(
            torch.sin(theta) * torch.sin(phi),
            torch.cos(dec_0_rad) * torch.cos(theta) -
            torch.sin(dec_0_rad) * torch.sin(theta) * torch.cos(phi)
        )


        return ra, dec

    def calculate_sourceCoords(self):
        """
        Calculate source coordinates from target image sky coordinates.
        """
        ra_rad, dec_rad = self.calculate_skyCoords()  # get RA,DEC in radians

        #ra_rad = torch.deg2rad(ra_rad)
        #dec_rad = torch.deg2rad(dec_rad)

        # Convert reference coordinates to radians
        ra_0_rad = torch.deg2rad(torch.tensor(self.source_wcs.RA_0, dtype=ra_rad.dtype))
        dec_0_rad = torch.deg2rad(torch.tensor(self.source_wcs.DEC_0, dtype=dec_rad.dtype))

        # Compute trigonometric terms
        cos_dec_0 = torch.cos(dec_0_rad)
        sin_dec_0 = torch.sin(dec_0_rad)
        cos_dec = torch.cos(dec_rad)
        sin_dec = torch.sin(dec_rad)

        # Compute RA difference, ensuring it's within [-pi, pi]
        d_ra = ra_rad - ra_0_rad
        d_ra = torch.where(d_ra > torch.pi, d_ra - 2 * torch.pi,
                           torch.where(d_ra < -torch.pi, d_ra + 2 * torch.pi, d_ra))

        # Compute cosine and sine terms
        cos_d_ra = torch.cos(d_ra)
        sin_d_ra = torch.sin(d_ra)

        # Compute denominator with numerical stability
        eps = 1e-10
        denom = sin_dec_0 * sin_dec + cos_dec_0 * cos_dec * cos_d_ra
        denom = torch.clamp(denom, min=eps, max=1 - eps)

        # Compute tangent plane coordinates
        x = -cos_dec * sin_d_ra / denom  # We need the negative sign here because of astronomical conventions
        y = (sin_dec * cos_dec_0 - cos_dec * sin_dec_0 * cos_d_ra) / denom

        # Scale coordinates
        x_scaled = x / self.source_wcs.CDELT1
        y_scaled = y / self.source_wcs.CDELT2


        # Apply inverse WCS transformation
        coords = torch.stack([x_scaled, y_scaled])
        coords_flat = coords.reshape(2, -1)
        transformed = torch.einsum('ij,jk->ik', self.source_wcs.inverse_PC_Matrix(), coords_flat)

        # Reshape back to original spatial dimensions
        x_intermediate = transformed[0].reshape(self.target_image.shape[0], self.target_image.shape[1])
        y_intermediate = transformed[1].reshape(self.target_image.shape[0], self.target_image.shape[1])

        # Apply inverse SIP distortion correction
        x_distorted = x_intermediate + self.source_wcs.SIP_polynomial_AP(x_intermediate,y_intermediate)
        y_distorted = y_intermediate + self.source_wcs.SIP_polynomial_BP(x_intermediate,y_intermediate)

        return x_distorted+ self.source_wcs.CRPIX1, y_distorted+ self.source_wcs.CRPIX2

    def interpolate_source_image(self, interpolation_mode='bilinear'):
        """
        Interpolate the source image at calculated source coordinates
        """
        # Get source coordinates
        x_source, y_source = self.calculate_sourceCoords()
        H, W = self.source_image.shape

        # Original grid_sample implementation for other modes
        x_normalized = 2.0 * (x_source / (W - 1)) - 1.0
        y_normalized = 2.0 * (y_source / (H - 1)) - 1.0

        # Stack coordinates into sampling grid
        grid = torch.stack([x_normalized, y_normalized], dim=-1)

        # Add batch and channel dimensions if needed
        source_image = self.source_image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        grid = grid.unsqueeze(0)  # [1, H, W, 2]

        # Perform interpolation
        resampled = torch.nn.functional.grid_sample(
            source_image,
            grid,
            mode=interpolation_mode,
            align_corners=True,
            padding_mode='zeros',
        )
        # Remove batch and channel dimensions
        resampled = resampled.squeeze()

        return resampled
def calculate_reprojection(source_hdu: fits.PrimaryHDU, target_hdu: fits.PrimaryHDU,  interpolation_mode='nearest'):
    """
    Wrapper function to calculate reprojection

    Args:
        target_hdu:
        source_hdu:
        interpolation_mode: mode of interpolation ['nearest', 'bilinear', 'bicubic']

    Returns:
        reprojected_image: tensor of reprojected image
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    target_wcs = WCSHeader.from_header(target_hdu.header)
    source_wcs = WCSHeader.from_header(source_hdu.header)
    # Convert the data to native byte order before creating tensors
    target_data = target_hdu.data.astype(target_hdu.data.dtype.newbyteorder('='))
    source_data = source_hdu.data.astype(source_hdu.data.dtype.newbyteorder('='))
    # Now create the tensors
    target_image = torch.tensor(target_data, dtype=torch.float32, device=device)
    source_image = torch.tensor(source_data, dtype=torch.float32, device=device)
    reprojection = Reproject(target_wcs=target_wcs, source_wcs=source_wcs, target_image=target_image, source_image=source_image, device=device)
    return reprojection.interpolate_source_image(interpolation_mode=interpolation_mode)

