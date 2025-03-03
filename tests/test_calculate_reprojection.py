import pytest
import torch
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import os

# Import the function to test
from reprojection.reproject import calculate_reprojection, Reproject


@pytest.mark.integration
class TestCalculateReprojection:
    """Tests for the high-level calculate_reprojection function."""

    @pytest.fixture
    def simple_source_target_pair(self, test_data_dir):
        """Create a simple source and target HDU pair for testing."""
        # Create a source HDU with a simple pattern
        source_wcs = WCS(naxis=2)
        source_wcs.wcs.crpix = [50, 50]
        source_wcs.wcs.cdelt = [0.1, 0.1]
        source_wcs.wcs.crval = [0, 0]
        source_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Create source data - Gaussian blob in the center
        source_data = np.zeros((100, 100), dtype=np.float32)
        y, x = np.mgrid[0:100, 0:100]
        source_data = 100.0 * np.exp(-((x - 50) ** 2 + (y - 50) ** 2) / 100.0)

        # Create source HDU
        source_hdu = fits.PrimaryHDU(data=source_data)
        source_hdu.header.update(source_wcs.to_header())

        # Create target HDU with a slightly different WCS
        target_wcs = WCS(naxis=2)
        target_wcs.wcs.crpix = [60, 60]  # Shifted reference pixel
        target_wcs.wcs.cdelt = [0.1, 0.1]
        target_wcs.wcs.crval = [0, 0]
        target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Create target data - empty array
        target_data = np.zeros((120, 120), dtype=np.float32)  # Larger dimensions

        # Create target HDU
        target_hdu = fits.PrimaryHDU(data=target_data)
        target_hdu.header.update(target_wcs.to_header())

        # Save HDUs to files for testing
        source_file = os.path.join(test_data_dir, "simple_source.fits")
        target_file = os.path.join(test_data_dir, "simple_target.fits")

        source_hdu.writeto(source_file, overwrite=True)
        target_hdu.writeto(target_file, overwrite=True)

        return source_hdu, target_hdu, source_file, target_file

    def test_basic_reprojection(self, simple_source_target_pair):
        """Test that the function performs basic reprojection correctly."""
        source_hdu, target_hdu, _, _ = simple_source_target_pair

        # Perform reprojection
        result = calculate_reprojection(
            source_hdu=source_hdu,
            target_hdu=target_hdu,
            interpolation_mode="bilinear"
        )

        # Check result properties
        assert isinstance(result, torch.Tensor)
        assert result.shape == target_hdu.data.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        # The result should have a gaussian peak somewhere
        assert result.max() > 0

    def test_all_interpolation_modes(self, simple_source_target_pair):
        """Test all supported interpolation modes."""
        source_hdu, target_hdu, _, _ = simple_source_target_pair

        modes = ["nearest", "bilinear", "bicubic"]

        for mode in modes:
            # Perform reprojection with each mode
            result = calculate_reprojection(
                source_hdu=source_hdu,
                target_hdu=target_hdu,
                interpolation_mode=mode
            )

            # Check basic properties
            assert isinstance(result, torch.Tensor)
            assert result.shape == target_hdu.data.shape
            assert not torch.isnan(result).any()

            # For this simple test case, all modes should produce valid results
            assert result.max() > 0

    def test_with_real_fits_files(self, source_fits_file, target_fits_file):
        """Test reprojection with real FITS files from fixtures."""
        # Load HDUs
        source_hdu = fits.open(source_fits_file)[0]
        target_hdu = fits.open(target_fits_file)[0]

        # Convert data to native byte order if needed
        if hasattr(source_hdu, 'data') and source_hdu.data is not None:
            source_hdu.data = np.asarray(source_hdu.data, dtype=np.float64).copy()

        if hasattr(target_hdu, 'data') and target_hdu.data is not None:
            target_hdu.data = np.asarray(target_hdu.data, dtype=np.float64).copy()

        # Perform reprojection
        result = calculate_reprojection(
            source_hdu=source_hdu,
            target_hdu=target_hdu,
            interpolation_mode="bilinear"
        )

        # Check result
        assert isinstance(result, torch.Tensor)
        assert result.shape == target_hdu.data.shape
        assert not torch.isnan(result).any()

    def test_end_to_end_workflow(self, simple_source_target_pair, test_data_dir):
        """Test a complete workflow including saving results to a new FITS file."""
        _, _, source_file, target_file = simple_source_target_pair

        # Load from files
        source_hdu = fits.open(source_file)[0]
        target_hdu = fits.open(target_file)[0]

        # Convert data to native byte order if needed
        if hasattr(source_hdu, 'data') and source_hdu.data is not None:
            source_hdu.data = np.asarray(source_hdu.data, dtype=np.float64).copy()

        if hasattr(target_hdu, 'data') and target_hdu.data is not None:
            target_hdu.data = np.asarray(target_hdu.data, dtype=np.float64).copy()

        # Perform reprojection
        reprojected = calculate_reprojection(
            source_hdu=source_hdu,
            target_hdu=target_hdu,
            interpolation_mode="bilinear"
        )

        # Convert to numpy and save as FITS
        reprojected_np = reprojected.cpu().numpy()
        output_hdu = fits.PrimaryHDU(data=reprojected_np, header=target_hdu.header)
        output_file = os.path.join(test_data_dir, "reprojected_output.fits")
        output_hdu.writeto(output_file, overwrite=True)

        # Verify the saved file exists and can be read
        assert os.path.exists(output_file)
        verification_hdu = fits.open(output_file)[0]
        assert verification_hdu.data.shape == target_hdu.data.shape

        # The data should match our reprojected tensor
        assert np.array_equal(verification_hdu.data, reprojected_np)

    def test_correct_reproject_instance_created(self, simple_source_target_pair, monkeypatch):
        """Test that the function creates a Reproject instance with the correct parameters."""
        source_hdu, target_hdu, _, _ = simple_source_target_pair

        # Create a mock to track how Reproject is called
        original_init = Reproject.__init__
        calls = []

        def mock_init(self, *args, **kwargs):
            calls.append((args, kwargs))
            return original_init(self, *args, **kwargs)

        # Apply the mock
        monkeypatch.setattr(Reproject, '__init__', mock_init)

        # Mock the interpolate_source_image method to return a dummy tensor
        original_interpolate = Reproject.interpolate_source_image

        def mock_interpolate(self, interpolation_mode="nearest"):
            # Return a dummy tensor with the right shape
            return torch.zeros(target_hdu.data.shape)

        monkeypatch.setattr(Reproject, 'interpolate_source_image', mock_interpolate)

        # Call the function
        result = calculate_reprojection(
            source_hdu=source_hdu,
            target_hdu=target_hdu,
            interpolation_mode="bilinear"
        )

        # Check that Reproject was initialized correctly
        assert len(calls) == 1
        init_args, init_kwargs = calls[0]

        # Check source_hdu and target_hdu were passed correctly
        assert init_kwargs.get('source_hdu') is source_hdu
        assert init_kwargs.get('target_hdu') is target_hdu