import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
from astropy.io.fits import PrimaryHDU, Header
from astropy.wcs import WCS

# Import the functions to test
from dfreproject.reproject import Reproject
from dfreproject.utils import (
    estimate_memory_per_pixel,
    calculate_chunk_size,
    process_chunk,
    reproject_chunked
)


class TestEstimateMemoryPerPixel(unittest.TestCase):
    """Test memory estimation per pixel."""
    
    def setUp(self):
        """Create a mock Reproject instance."""
        self.mock_reproject = Mock(spec=Reproject)
        self.mock_reproject.device = torch.device('cpu')
        
    def test_memory_estimate_single_batch(self):
        """Test memory estimation for single batch."""
        self.mock_reproject.batch_source_images = [torch.zeros(100, 100)]
        self.mock_reproject.conserve_flux = True
        self.mock_reproject.compute_jacobian = False
        
        memory = estimate_memory_per_pixel(self.mock_reproject, "bilinear")
        
        # Base calculation: (2+2+2+1+1)*8 = 64 bytes per pixel
        # Times batch size (1): 64
        # Times overhead (1.2): 76.8
        expected_base = 64 * 1 * 1.2
        self.assertAlmostEqual(memory, expected_base, delta=1.0)
    
    def test_memory_estimate_with_jacobian(self):
        """Test memory estimation includes Jacobian overhead."""
        self.mock_reproject.batch_source_images = [torch.zeros(100, 100)]
        self.mock_reproject.conserve_flux = True
        self.mock_reproject.compute_jacobian = True
        
        memory = estimate_memory_per_pixel(self.mock_reproject, "bilinear")
        
        # Base: 64 bytes, Jacobian adds 4*8=32 bytes
        # Total: (64+32)*1*1.2 = 115.2
        expected_with_jacobian = (64 + 32) * 1 * 1.2
        self.assertAlmostEqual(memory, expected_with_jacobian, delta=1.0)
    
    def test_memory_estimate_multiple_batches(self):
        """Test memory estimation scales with batch size."""
        self.mock_reproject.batch_source_images = [
            torch.zeros(100, 100),
            torch.zeros(100, 100),
            torch.zeros(100, 100)
        ]
        self.mock_reproject.conserve_flux = True
        self.mock_reproject.compute_jacobian = False
        
        memory = estimate_memory_per_pixel(self.mock_reproject, "bilinear")
        
        # Base: 64 bytes * 3 batches * 1.2 overhead = 230.4
        expected_batched = 64 * 3 * 1.2
        self.assertAlmostEqual(memory, expected_batched, delta=1.0)
    
    def test_memory_estimate_no_flux_conservation(self):
        """Test memory estimation without flux conservation."""
        self.mock_reproject.batch_source_images = [torch.zeros(100, 100)]
        self.mock_reproject.conserve_flux = False
        self.mock_reproject.compute_jacobian = False
        
        memory = estimate_memory_per_pixel(self.mock_reproject, "bilinear")
        
        # Should still calculate base memory
        self.assertGreater(memory, 0)


class TestCalculateChunkSize(unittest.TestCase):
    """Test chunk size calculation."""
    
    def setUp(self):
        """Create a mock Reproject instance."""
        self.mock_reproject = Mock(spec=Reproject)
        self.mock_reproject.device = torch.device('cpu')
        self.mock_reproject.batch_source_images = [torch.zeros(100, 100)]
        self.mock_reproject.conserve_flux = True
        self.mock_reproject.compute_jacobian = False
    
    def test_chunk_size_with_ample_memory(self):
        """Test chunk size calculation with plenty of memory."""
        output_shape = (1000, 1000)
        max_memory_mb = 1000.0  # 1GB
        safety_factor = 0.8
        
        chunk_h, chunk_w = calculate_chunk_size(
            self.mock_reproject,
            output_shape,
            max_memory_mb,
            safety_factor,
            "bilinear"
        )
        
        # Should be able to process entire image
        self.assertGreater(chunk_h, 0)
        self.assertGreater(chunk_w, 0)
        self.assertLessEqual(chunk_h, output_shape[0])
        self.assertLessEqual(chunk_w, output_shape[1])
    
    def test_chunk_size_with_tight_memory(self):
        """Test chunk size calculation with limited memory."""
        output_shape = (4000, 4000)
        max_memory_mb = 10.0  # Very limited
        safety_factor = 0.8
        
        chunk_h, chunk_w = calculate_chunk_size(
            self.mock_reproject,
            output_shape,
            max_memory_mb,
            safety_factor,
            "bilinear"
        )
        
        # Should create smaller chunks
        self.assertGreater(chunk_h, 0)
        self.assertGreater(chunk_w, 0)
        self.assertLess(chunk_h * chunk_w, output_shape[0] * output_shape[1])
    
    def test_chunk_size_minimum_one_row(self):
        """Test that chunk size is at least one row."""
        output_shape = (1000, 1000)
        max_memory_mb = 0.001  # Extremely limited
        safety_factor = 0.8
        
        with patch('dfreproject.reproject.logger') as mock_logger:
            chunk_h, chunk_w = calculate_chunk_size(
                self.mock_reproject,
                output_shape,
                max_memory_mb,
                safety_factor,
                "bilinear"
            )

            # Should still return valid dimensions
            self.assertGreaterEqual(chunk_h, 1)
            self.assertGreaterEqual(chunk_w, 1)
    
    def test_chunk_size_respects_image_bounds(self):
        """Test that chunk size doesn't exceed image dimensions."""
        output_shape = (100, 200)
        max_memory_mb = 10000.0  # Huge memory
        safety_factor = 0.8
        
        chunk_h, chunk_w = calculate_chunk_size(
            self.mock_reproject,
            output_shape,
            max_memory_mb,
            safety_factor,
            "bilinear"
        )
        
        # Chunks should not exceed image dimensions
        self.assertLessEqual(chunk_h, output_shape[0])
        self.assertLessEqual(chunk_w, output_shape[1])
    
    def test_chunk_size_roughly_square(self):
        """Test that chunks are approximately square when possible."""
        output_shape = (1000, 1000)
        max_memory_mb = 50.0
        safety_factor = 0.8
        
        chunk_h, chunk_w = calculate_chunk_size(
            self.mock_reproject,
            output_shape,
            max_memory_mb,
            safety_factor,
            "bilinear"
        )
        
        # Aspect ratio should be reasonably close to 1
        aspect_ratio = chunk_h / chunk_w
        self.assertGreater(aspect_ratio, 0.5)
        self.assertLess(aspect_ratio, 2.0)


class TestProcessChunk(unittest.TestCase):
    """Test single chunk processing."""
    
    def setUp(self):
        """Create a simple test setup."""
        # Create simple WCS for testing
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.crpix = [50, 50]
        self.wcs.wcs.crval = [0, 0]
        self.wcs.wcs.cdelt = [0.1, 0.1]
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        # Create test data
        self.data = np.ones((100, 100), dtype=np.float64)
        self.header = self.wcs.to_header()
        self.hdu = PrimaryHDU(data=self.data, header=self.header)
    
    def test_process_chunk_returns_correct_shape(self):
        """Test that process_chunk returns correct output shape."""
        shape_out = (100, 100)
        reproject = Reproject(
            source_hdus=[self.hdu],
            target_wcs=self.wcs,
            shape_out=shape_out,
            device='cpu'
        )
        
        # Process a chunk
        y_start, y_end = 0, 50
        x_start, x_end = 0, 50
        
        result = process_chunk(
            reproject,
            y_start, y_end,
            x_start, x_end,
            interpolation_mode="nearest"
        )
        
        # Check shape
        expected_shape = (y_end - y_start, x_end - x_start)
        self.assertEqual(result.shape, expected_shape)
    
    def test_process_chunk_restores_grid(self):
        """Test that process_chunk restores original grid after processing."""
        shape_out = (100, 100)
        reproject = Reproject(
            source_hdus=[self.hdu],
            target_wcs=self.wcs,
            shape_out=shape_out,
            device='cpu'
        )
        
        original_grid = reproject.target_grid
        
        # Process a chunk
        process_chunk(reproject, 0, 50, 0, 50, "nearest")
        
        # Grid should be restored
        self.assertIs(reproject.target_grid, original_grid)
    
    def test_process_chunk_different_regions(self):
        """Test processing different chunk regions."""
        shape_out = (100, 100)
        reproject = Reproject(
            source_hdus=[self.hdu],
            target_wcs=self.wcs,
            shape_out=shape_out,
            device='cpu'
        )
        
        # Process different chunks
        chunk1 = process_chunk(reproject, 0, 25, 0, 25, "nearest")
        chunk2 = process_chunk(reproject, 25, 50, 25, 50, "nearest")
        
        # Both should return valid tensors
        self.assertIsInstance(chunk1, torch.Tensor)
        self.assertIsInstance(chunk2, torch.Tensor)
        self.assertEqual(chunk1.shape, (25, 25))
        self.assertEqual(chunk2.shape, (25, 25))
    
    @patch('torch.cuda.empty_cache')
    def test_process_chunk_clears_cache_on_cuda(self, mock_empty_cache):
        """Test that CUDA cache is cleared after chunk processing."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        shape_out = (100, 100)
        reproject = Reproject(
            source_hdus=[self.hdu],
            target_wcs=self.wcs,
            shape_out=shape_out,
            device='cuda'
        )
        
        process_chunk(reproject, 0, 50, 0, 50, "nearest")
        
        # Cache should be cleared
        mock_empty_cache.assert_called()


class TestReprojectChunked(unittest.TestCase):
    """Test full chunked reprojection."""
    
    def setUp(self):
        """Create a simple test setup."""
        # Create simple WCS
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.crpix = [50, 50]
        self.wcs.wcs.crval = [0, 0]
        self.wcs.wcs.cdelt = [0.1, 0.1]
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        # Create test data
        self.data = np.random.rand(100, 100).astype(np.float64)
        self.header = self.wcs.to_header()
        self.hdu = PrimaryHDU(data=self.data, header=self.header)
    
    def test_reproject_chunked_returns_correct_shape(self):
        """Test that chunked reprojection returns correct output shape."""
        shape_out = (100, 100)
        reproject = Reproject(
            source_hdus=[self.hdu],
            target_wcs=self.wcs,
            shape_out=shape_out,
            device='cpu'
        )
        
        result = reproject_chunked(
            reproject,
            max_memory_mb=10.0,
            safety_factor=0.8,
            interpolation_mode="nearest",
            show_progress=False
        )
        
        # Check output shape
        expected_shape = (1, 100, 100)
        self.assertEqual(result.shape, expected_shape)
    
    def test_reproject_chunked_handles_single_chunk(self):
        """Test chunked reprojection with single chunk (ample memory)."""
        shape_out = (50, 50)
        reproject = Reproject(
            source_hdus=[self.hdu],
            target_wcs=self.wcs,
            shape_out=shape_out,
            device='cpu'
        )
        
        result = reproject_chunked(
            reproject,
            max_memory_mb=1000.0,  # Plenty of memory
            safety_factor=0.8,
            interpolation_mode="nearest",
            show_progress=False
        )
        
        # Should still work correctly
        self.assertEqual(result.shape, (1, 50, 50))
    
    def test_reproject_chunked_handles_multiple_chunks(self):
        """Test chunked reprojection with multiple chunks."""
        shape_out = (100, 100)
        reproject = Reproject(
            source_hdus=[self.hdu],
            target_wcs=self.wcs,
            shape_out=shape_out,
            device='cpu'
        )
        
        result = reproject_chunked(
            reproject,
            max_memory_mb=1.0,  # Force multiple chunks
            safety_factor=0.8,
            interpolation_mode="nearest",
            show_progress=False
        )
        
        # Should produce complete output
        self.assertEqual(result.shape, (1, 100, 100))
        
        # Should not be all NaN (some valid data)
        self.assertFalse(torch.isnan(result).all())
    
    
    
    def test_reproject_chunked_with_multiple_batches(self):
        """Test chunked reprojection with multiple source images."""
        shape_out = (100, 100)
        hdus = [self.hdu, self.hdu]  # Two identical HDUs
        
        reproject = Reproject(
            source_hdus=hdus,
            target_wcs=self.wcs,
            shape_out=shape_out,
            device='cpu'
        )
        
        result = reproject_chunked(
            reproject,
            max_memory_mb=10.0,
            safety_factor=0.8,
            interpolation_mode="nearest",
            show_progress=False
        )
        
        # Should have 2 batches
        self.assertEqual(result.shape, (2, 100, 100))
    
    
    
    def test_reproject_chunked_different_interpolation_modes(self):
        """Test chunked reprojection with different interpolation modes."""
        shape_out = (100, 100)
        reproject = Reproject(
            source_hdus=[self.hdu],
            target_wcs=self.wcs,
            shape_out=shape_out,
            device='cpu'
        )
        
        for mode in ["nearest", "bilinear"]:
            with self.subTest(mode=mode):
                result = reproject_chunked(
                    reproject,
                    max_memory_mb=10.0,
                    safety_factor=0.8,
                    interpolation_mode=mode,
                    show_progress=False
                )
                
                self.assertEqual(result.shape, (1, 100, 100))


class TestChunkedIntegration(unittest.TestCase):
    """Integration tests for chunked reprojection."""
    
    def setUp(self):
        """Create test data for integration tests."""
        # Create source WCS
        self.source_wcs = WCS(naxis=2)
        self.source_wcs.wcs.crpix = [50, 50]
        self.source_wcs.wcs.crval = [0, 0]
        self.source_wcs.wcs.cdelt = [0.1, 0.1]
        self.source_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        # Create target WCS (slightly offset)
        self.target_wcs = WCS(naxis=2)
        self.target_wcs.wcs.crpix = [50, 50]
        self.target_wcs.wcs.crval = [0.5, 0.5]
        self.target_wcs.wcs.cdelt = [0.1, 0.1]
        self.target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        # Create test image with pattern
        x, y = np.meshgrid(np.arange(100), np.arange(100))
        self.data = (x + y).astype(np.float64)
        self.header = self.source_wcs.to_header()
        self.hdu = PrimaryHDU(data=self.data, header=self.header)
    
    def test_end_to_end_chunked_vs_normal(self):
        """Test that chunked processing produces same result as normal."""
        from dfreproject.reproject import calculate_reprojection
        
        shape_out = (100, 100)
        
        # Normal processing
        result_normal = calculate_reprojection(
            source_hdus=self.hdu,
            target_wcs=self.target_wcs,
            shape_out=shape_out,
            order="nearest",
            device='cpu',
            conserve_flux=False,
            compute_jacobian=False
        )
        
        # Chunked processing
        result_chunked = calculate_reprojection(
            source_hdus=self.hdu,
            target_wcs=self.target_wcs,
            shape_out=shape_out,
            order="nearest",
            device='cpu',
            conserve_flux=False,
            compute_jacobian=False,
            max_memory_mb=5.0  # Force chunking
        )
        
        # Results should be nearly identical
        # Use numpy for comparison since results are arrays
        mask = ~np.isnan(result_normal) & ~np.isnan(result_chunked)
        if mask.any():
            diff = np.abs(result_normal[mask] - result_chunked[mask])
            self.assertLess(np.max(diff), 1e-6)


if __name__ == '__main__':
    unittest.main()