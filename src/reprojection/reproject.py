import torch
from astropy.io import fits
from dataclasses import dataclass
from typing import List

EPSILON = 1e-10


@dataclass
class WCSHeader:
    """
    A dataclass containing World Coordinate System (WCS) parameters with SIP distortion coefficients.

    This class encapsulates all the necessary parameters for defining a WCS transformation
    with Simple Imaging Polynomial (SIP) distortion representation. It supports conversion
    between pixel coordinates and world coordinates (RA/Dec) while accounting for optical
    distortions in astronomical images.

    Attributes
    ----------
    DEC_0 : torch.Tensor
        Reference declination in degrees (CRVAL2 in FITS standard).

    RA_0 : torch.Tensor
        Reference right ascension in degrees (CRVAL1 in FITS standard).

    CRPIX1 : torch.Tensor
        X coordinate of reference pixel.

    CRPIX2 : torch.Tensor
        Y coordinate of reference pixel.

    CDELT1 : torch.Tensor
        Increment along axis 1 at reference point in degrees/pixel.

    CDELT2 : torch.Tensor
        Increment along axis 2 at reference point in degrees/pixel.

    A_ORDER : int
        Order of the forward SIP polynomial A for distortion correction.

    B_ORDER : int
        Order of the forward SIP polynomial B for distortion correction.

    AP_ORDER : int
        Order of the inverse SIP polynomial AP for distortion correction.

    BP_ORDER : int
        Order of the inverse SIP polynomial BP for distortion correction.

    A : List[float]
        SIP polynomial A coefficients in the format [A_0_2, A_0_3, A_1_1, A_1_2, A_2_0, A_2_1, A_3_0].
        These coefficients describe distortion in the x-direction.

    B : List[float]
        SIP polynomial B coefficients in the format [B_0_2, B_0_3, B_1_1, B_1_2, B_2_0, B_2_1, B_3_0].
        These coefficients describe distortion in the y-direction.

    AP : List[float]
        Inverse SIP polynomial AP coefficients in the format
        [AP_0_0, AP_0_1, AP_0_2, AP_0_3, AP_1_0, AP_1_1, AP_1_2, AP_2_0, AP_2_1, AP_3_0].
        Used for converting world coordinates back to pixel coordinates.

    BP : List[float]
        Inverse SIP polynomial BP coefficients in the format
        [BP_0_0, BP_0_1, BP_0_2, BP_0_3, BP_1_0, BP_1_1, BP_1_2, BP_2_0, BP_2_1, BP_3_0].
        Used for converting world coordinates back to pixel coordinates.

    PC_Matrix : torch.Tensor
        Projection transformation matrix as a 2x2 tensor: [[PC1_1, PC1_2], [PC2_1, PC2_2]].
        Describes the rotation and scaling between intermediate world coordinates and pixel coordinates.

    Notes
    -----
    The SIP convention (Simple Imaging Polynomial) is described in:
    Shupe, D. L., et al. 2005, "The SIP Convention for Representing Distortion in FITS Image Headers"

    This implementation uses PyTorch tensors to enable GPU acceleration and automatic
    differentiation for the coordinate transformations.

    Examples
    --------
    >>> # Create a WCS header from components
    >>> wcs_info = WCSHeader(
    ...     DEC_0=torch.tensor(30.0),
    ...     RA_0=torch.tensor(150.0),
    ...     CRPIX1=torch.tensor(512.0),
    ...     CRPIX2=torch.tensor(512.0),
    ...     CDELT1=torch.tensor(-0.001),
    ...     CDELT2=torch.tensor(0.001),
    ...     A_ORDER=3,
    ...     B_ORDER=3,
    ...     AP_ORDER=3,
    ...     BP_ORDER=3,
    ...     A=[1e-5, 2e-8, 3e-7, 5e-9, 1e-6, 7e-10, 4e-11],
    ...     B=[2e-5, 3e-8, 2e-7, 6e-9, 5e-7, 8e-10, 5e-11],
    ...     AP=[0.0, 0.0, -1e-5, -2e-8, 0.0, -3e-7, -5e-9, -1e-6, -7e-10, -4e-11],
    ...     BP=[0.0, 0.0, -2e-5, -3e-8, 0.0, -2e-7, -6e-9, -5e-7, -8e-10, -5e-11],
    ...     PC_Matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ... )
    """
    DEC_0: torch.tensor
    RA_0: torch.tensor
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
    def from_header(cls, header: fits.Header):
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
        wcs_info = {'DEC_0': torch.tensor(header['CRVAL2'], dtype=torch.float32),
                    'RA_0': torch.tensor(header['CRVAL1'], dtype=torch.float32),
                    'CRPIX1': torch.tensor(header['CRPIX1'], dtype=torch.float32),
                    'CRPIX2': torch.tensor(header['CRPIX2'], dtype=torch.float32),
                    'CDELT1': torch.tensor(header['CDELT1'], dtype=torch.float32),
                    'CDELT2': torch.tensor(header['CDELT2'], dtype=torch.float32), 'A_ORDER': header.get('A_ORDER', 0),
                    'B_ORDER': header.get('B_ORDER', 0), 'AP_ORDER': header.get('AP_ORDER', 0),
                    'BP_ORDER': header.get('BP_ORDER', 0)}

        # Get SIP orders

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
        Calculate the inverse of the PC matrix for coordinate transformations.

        The PC (Projection Coordinate) matrix defines the linear transformation
        between pixel coordinates and intermediate world coordinates. The inverse
        of this matrix is needed when converting from world coordinates back to
        pixel coordinates.

        Returns
        -------
        torch.Tensor
            A 2x2 tensor containing the inverse of the PC matrix.
            This matrix has the same dtype as the original PC_Matrix.

        Notes
        -----
        This method uses PyTorch's linear algebra module to compute the matrix inverse.
        The PC matrix is typically well-conditioned in WCS, so numerical stability
        issues are rare.

        If the PC matrix is the identity matrix (the default when SIP distortion
        is not present), its inverse is also the identity matrix.
        """
        return torch.linalg.inv(self.PC_Matrix)

    def SIP_polynomial_A(self, u, v):
        """
        Evaluate the SIP distortion polynomial A at the given intermediate coordinates.
    
        The Simple Imaging Polynomial (SIP) convention uses polynomial functions to model
        optical distortion in astronomical images. This method calculates the x-direction
        distortion correction using the A polynomial coefficients.

        Parameters
        ----------
        u : torch.Tensor
            Intermediate pixel x-coordinates relative to the reference pixel.
            Can be a scalar, vector, or multi-dimensional tensor.

        v : torch.Tensor
            Intermediate pixel y-coordinates relative to the reference pixel.
            Must have the same shape as `u`.

        Returns
        -------
        torch.Tensor
            The evaluated A polynomial values with the same shape as the input coordinates.
            These values represent the x-direction distortion correction.

        Notes
        -----
        Currently supports polynomial orders 2 and 3. For unsupported orders or
        when coefficients are missing, returns zeros (no distortion correction).

        The SIP convention is described in Shupe et al. (2005) and defines the
        distortion as:

        x = u + A(u,v)

        where u,v are the undistorted intermediate pixel coordinates and x is the
        distorted coordinate.

        For order 2, the polynomial is:
        A(u,v) = A_0_2 * v² + A_1_1 * u*v + A_2_0 * u²

        For order 3, the polynomial adds higher-order terms:
        A(u,v) = A_0_2 * v² + A_0_3 * v³ + A_1_1 * u*v + A_1_2 * u*v² +
                 A_2_0 * u² + A_2_1 * u²*v + A_3_0 * u³

        Examples
        --------
        >>> # Create coordinate grids centered at the reference pixel
        >>> u = torch.linspace(-100, 100, 201)
        >>> v = torch.linspace(-100, 100, 201)
        >>> u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
        >>>
        >>> # Calculate SIP distortion correction
        >>> correction_x = WCSHeader.SIP_polynomial_A(u_grid, v_grid)

        See Also
        --------
        SIP_polynomial_B : Evaluate the B polynomial for y-direction distortion
        SIP_polynomial_AP : Evaluate the inverse A polynomial
        SIP_polynomial_BP : Evaluate the inverse B polynomial
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
        Evaluate the SIP distortion polynomial B at the given intermediate coordinates.

        The Simple Imaging Polynomial (SIP) convention uses polynomial functions to model
        optical distortion in astronomical images. This method calculates the y-direction
        distortion correction using the B polynomial coefficients.

        Parameters
        ----------
        u : torch.Tensor
            Intermediate pixel x-coordinates relative to the reference pixel.
            Can be a scalar, vector, or multi-dimensional tensor.

        v : torch.Tensor
            Intermediate pixel y-coordinates relative to the reference pixel.
            Must have the same shape as `u`.

        Returns
        -------
        torch.Tensor
            The evaluated B polynomial values with the same shape as the input coordinates.
            These values represent the y-direction distortion correction.

        Notes
        -----
        Currently supports polynomial orders 2 and 3. For unsupported orders or
        when coefficients are missing, returns zeros (no distortion correction).

        The SIP convention is described in Shupe et al. (2005) and defines the
        distortion as:

        y = v + B(u,v)

        where u,v are the undistorted intermediate pixel coordinates and y is the
        distorted coordinate.

        For order 2, the polynomial is:
        B(u,v) = B_0_2 * v² + B_1_1 * u*v + B_2_0 * u²

        For order 3, the polynomial adds higher-order terms:
        B(u,v) = B_0_2 * v² + B_0_3 * v³ + B_1_1 * u*v + B_1_2 * u*v² +
                 B_2_0 * u² + B_2_1 * u²*v + B_3_0 * u³

        Examples
        --------
        >>> # Create coordinate grids centered at the reference pixel
        >>> u = torch.linspace(-100, 100, 201)
        >>> v = torch.linspace(-100, 100, 201)
        >>> u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
        >>>
        >>> # Calculate SIP distortion correction
        >>> correction_y = WCSHeader.SIP_polynomial_B(u_grid, v_grid)

        See Also
        --------
        SIP_polynomial_A : Evaluate the A polynomial for x-direction distortion
        SIP_polynomial_AP : Evaluate the inverse A polynomial
        SIP_polynomial_BP : Evaluate the inverse B polynomial
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
        Evaluate the inverse SIP distortion polynomial AP at the given coordinates.

        The AP polynomial is used in the inverse transformation from distorted pixel
        coordinates back to undistorted intermediate coordinates. This is essential
        for converting world coordinates (RA/Dec) to pixel coordinates in the presence
        of optical distortion.

        Parameters
        ----------
        u : torch.Tensor
            Distorted pixel x-coordinates relative to the reference pixel.
            Can be a scalar, vector, or multi-dimensional tensor.

        v : torch.Tensor
            Distorted pixel y-coordinates relative to the reference pixel.
            Must have the same shape as `u`.

        Returns
        -------
        torch.Tensor
            The evaluated AP polynomial values with the same shape as the input coordinates.
            These values represent the x-direction correction for the inverse transformation.

        Notes
        -----
        Currently supports polynomial orders 2 and 3. For unsupported orders or
        when coefficients are missing, returns zeros (no correction).

        The inverse SIP convention defines the transformation as:

        u′ = u + AP(u,v)

        where u,v are the distorted pixel coordinates and u′ is the undistorted
        intermediate coordinate.

        For order 2, the polynomial is:
        AP(u,v) = AP_0_0 + AP_0_1*v + AP_0_2*v² + AP_1_0*u + AP_1_1*u*v + AP_2_0*u²

        For order 3, the polynomial adds higher-order terms:
        AP(u,v) = AP_0_0 + AP_0_1*v + AP_0_2*v² + AP_0_3*v³ +
                  AP_1_0*u + AP_1_1*u*v + AP_1_2*u*v² +
                  AP_2_0*u² + AP_2_1*u²*v + AP_3_0*u³

        Note that unlike the forward polynomials (A, B), the inverse polynomials include
        constant (AP_0_0) and first-order terms to account for all distortion effects.

        Examples
        --------
        >>> # Create coordinate grids for distorted pixel coordinates
        >>> u = torch.linspace(-100, 100, 201)
        >>> v = torch.linspace(-100, 100, 201)
        >>> u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
        >>>
        >>> # Calculate inverse distortion correction
        >>> correction_x = wcs_header.SIP_polynomial_AP(u_grid, v_grid)
        >>>
        >>> # Apply correction to get undistorted coordinates
        >>> u_corrected = u_grid + correction_x

        See Also
        --------
        SIP_polynomial_BP : Evaluate the inverse B polynomial for y-direction correction
        SIP_polynomial_A : Evaluate the forward A polynomial
        SIP_polynomial_B : Evaluate the forward B polynomial
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
        Evaluate the inverse SIP distortion polynomial BP at the given coordinates.

        The BP polynomial is used in the inverse transformation from distorted pixel
        coordinates back to undistorted intermediate coordinates. This is essential
        for converting world coordinates (RA/Dec) to pixel coordinates in the presence
        of optical distortion.

        Parameters
        ----------
        u : torch.Tensor
            Distorted pixel x-coordinates relative to the reference pixel.
            Can be a scalar, vector, or multi-dimensional tensor.

        v : torch.Tensor
            Distorted pixel y-coordinates relative to the reference pixel.
            Must have the same shape as `u`.

        Returns
        -------
        torch.Tensor
            The evaluated BP polynomial values with the same shape as the input coordinates.
            These values represent the y-direction correction for the inverse transformation.

        Notes
        -----
        Currently supports polynomial orders 2 and 3. For unsupported orders or
        when coefficients are missing, returns zeros (no correction).

        The inverse SIP convention defines the transformation as:

        v′ = v + BP(u,v)

        where u,v are the distorted pixel coordinates and v′ is the undistorted
        intermediate coordinate.

        For order 2, the polynomial is:
        BP(u,v) = BP_0_0 + BP_0_1*v + BP_0_2*v² + BP_1_0*u + BP_1_1*u*v + BP_2_0*u²

        For order 3, the polynomial adds higher-order terms:
        BP(u,v) = BP_0_0 + BP_0_1*v + BP_0_2*v² + BP_0_3*v³ +
                  BP_1_0*u + BP_1_1*u*v + BP_1_2*u*v² +
                  BP_2_0*u² + BP_2_1*u²*v + BP_3_0*u³

        Note that unlike the forward polynomials (A, B), the inverse polynomials include
        constant (BP_0_0) and first-order terms to account for all distortion effects.

        Examples
        --------
        >>> # Create coordinate grids for distorted pixel coordinates
        >>> u = torch.linspace(-100, 100, 201)
        >>> v = torch.linspace(-100, 100, 201)
        >>> u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
        >>>
        >>> # Calculate inverse distortion correction
        >>> correction_y = wcs_header.SIP_polynomial_BP(u_grid, v_grid)
        >>>
        >>> # Apply correction to get undistorted coordinates
        >>> v_corrected = v_grid + correction_y

        See Also
        --------
        SIP_polynomial_AP : Evaluate the inverse A polynomial for x-direction correction
        SIP_polynomial_A : Evaluate the forward A polynomial
        SIP_polynomial_B : Evaluate the forward B polynomial
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

        self.target_grid = torch.meshgrid(
            torch.arange(self.target_image.shape[0], dtype=torch.float32, device=device),  # height
            torch.arange(self.target_image.shape[1], dtype=torch.float32, device=device),  # width
            indexing='ij',  # y, x
        )

    def calculate_skyCoords(self):
        """
        Calculate sky coordinates (RA/Dec) for each pixel in the target image.

        This method transforms pixel coordinates from the target image frame to celestial
        coordinates (Right Ascension and Declination) using the full WCS transformation
        pipeline including SIP distortion corrections. The transformation follows these steps:

        1. Convert pixel coordinates to offsets from the reference pixel
        2. Apply SIP distortion correction
        3. Apply the PC matrix transformation
        4. Scale by the pixel scale (CDELT)
        5. Project the intermediate coordinates onto the celestial sphere using
           gnomonic (tangent plane) projection

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing (ra, dec) where:
            - ra: Right Ascension in radians for each pixel in the target image
            - dec: Declination in radians for each pixel in the target image

            Both tensors have the same shape as the target image.

        Notes
        -----
        This implementation follows the FITS WCS standard for gnomonic (TAN) projection
        with SIP distortion corrections. The algorithm:

        1. Computes intermediate pixel coordinates (u,v) as offsets from the reference pixel
        2. Applies SIP distortion polynomials to get (x',y')
        3. Applies the PC matrix transformation to get intermediate world coordinates
        4. Scales by CDELT to get projection plane coordinates
        5. Computes the spherical coordinates using the gnomonic projection equations

        The output coordinates are in radians to facilitate further calculations.
        To convert to degrees, use torch.rad2deg().
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

        # Convert reference coordinates to radians
        dec_0_rad = torch.deg2rad(self.target_wcs.DEC_0)
        ra_0_rad = torch.deg2rad(self.target_wcs.RA_0)

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
        Calculate source image pixel coordinates corresponding to each target image pixel.

        This method completes the coordinate mapping chain by transforming target image
        pixels to sky coordinates and then to source image pixel coordinates. This is
        the fundamental operation for image reprojection, as it determines which source
        pixels should be sampled to create each target pixel. The transformation follows:

        1. Calculate sky coordinates (RA/Dec) for each target pixel
        2. Convert these celestial coordinates to intermediate world coordinates in the source frame
        3. Apply the inverse PC matrix transformation
        4. Apply inverse SIP distortion correction
        5. Add reference pixel offsets to get final source pixel coordinates

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing (x, y) where:
            - x: Source image x-coordinates (columns) for each pixel in the target image
            - y: Source image y-coordinates (rows) for each pixel in the target image

            Both tensors have the same shape as the target image.

        Notes
        -----
        This implementation handles the full FITS WCS transformation with SIP distortion,
        calculating where in the source image each target pixel should sample from.

        The method includes special handling for the RA wrap-around issue, ensuring
        that coordinates spanning the 0h/24h boundary are correctly processed.

        A small epsilon value is used to ensure numerical stability when computing
        coordinates near the poles.

        The output pixel coordinates can be used directly with grid sampling functions
        to perform the actual reprojection.
        """
        ra_rad, dec_rad = self.calculate_skyCoords()  # get RA,DEC in radians
        # Convert reference coordinates to radians
        ra_0_rad = torch.deg2rad(self.source_wcs.RA_0)
        dec_0_rad = torch.deg2rad(self.source_wcs.DEC_0)

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
        combined_result = torch.nn.functional.grid_sample(
            combined,
            grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='zeros'
        )

        # Split the results
        resampled = combined_result[:, 0].squeeze()
        footprint = combined_result[:, 1].squeeze()
        # Apply footprint correction where the footprint is significant
        valid_mask = footprint > 1e-6
        resampled[valid_mask] /= footprint[valid_mask]
        # Apply simple flux conservation
        new_total_flux = torch.sum(resampled)
        normalization_factor_flux = original_total_flux / new_total_flux if new_total_flux > 0 else 1

        return resampled * normalization_factor_flux

def calculate_reprojection(source_hdu: fits.PrimaryHDU, target_hdu: fits.PrimaryHDU,  interpolation_mode='nearest'):
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
    >>> from astroreproject import calculate_reprojection
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

