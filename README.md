# Astronomical Image Reprojection

A high-performance Python package for reprojecting astronomical images between different coordinate systems with support for SIP distortion correction.

[![Documentation Status](https://readthedocs.org/projects/reprojection/badge/?version=latest)](https://reprojection.readthedocs.io/en/latest/?badge=latest)


## Features

- Fast reprojection of astronomical images between different WCS frames
- Support for Simple Imaging Polynomial (SIP) distortion correction
- GPU acceleration using PyTorch
- Multiple interpolation methods (nearest neighbor, bilinear, bicubic)
- Simple high-level API and detailed low-level control

## Installation

### Requirements

- Python 3.7+
- NumPy
- Astropy
- PyTorch
- Matplotlib (for visualization)
- cmcrameri

### Installing from Source

For the latest development version, install directly from the GitHub repository:

```bash
git clone https://github.com/dragonfly/reprojection.git
cd reprojection
pip install -e .
```

For development installation with documentation dependencies:

```bash
pip install -e ".[docs]"
```

## Quick Start

```python
from astropy.io import fits
from reprojection import calculate_reprojection

# Load source and target images
source_hdu = fits.open('source_image.fits')[0]
target_hdu = fits.open('target_grid.fits')[0]

# Perform reprojection with bilinear interpolation
reprojected = calculate_reprojection(
    source_hdu=source_hdu, 
    target_hdu=target_hdu,
    interpolation_mode='bilinear'
)

# Convert back to NumPy and save as FITS
reprojected_np = reprojected.cpu().numpy()
output_hdu = fits.PrimaryHDU(data=reprojected_np, header=target_hdu.header)
output_hdu.writeto('reprojected_image.fits', overwrite=True)
```

## Demos and Examples

A collection of example notebooks and scripts is available in the `demos` folder to help you get started:

- `demo.ipynb` - Simple example of reprojecting between two WCS frames

To run the demos:

```bash
cd demos
jupyter notebook
```

## Documentation

Comprehensive documentation is available at [https://reprojection.readthedocs.io/](https://reprojection.readthedocs.io/)

The documentation includes:
- API reference
- Mathematical details of the reprojection process
- Tutorials and examples
- Performance tips

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:



## Acknowledgments

- Based on the FITS WCS standard and SIP convention
- Inspired by Astropy's reproject package
- Accelerated with PyTorch
- Documentation aided by Claude.ai
