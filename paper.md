---
title: 'dfreproject: A Python package for astronomical reprojection'
tags:
  - Python
  - astronomy
  - reprojection
authors:
  - name: Carter Lee Rhea
    orcid: 0000-0003-2001-1076
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Pieter Van Dokkum
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Steven R. Janssens
    orcid: 0000-0003-0327-3322
    affiliation: 1
  - name: Imad Pasha
    affiliation: 1
  - name: Roberto Abraham
    affiliation: 1

affiliations:
 - name: Dragonfly Focused Research Organization, 150 Washington Avenue, Santa Fe, 87501, NM, USA
   index: 1
 - name: Centre de recherche en astrophysique du Québec (CRAQ)
   index: 2
date: 01 April 2025
bibliography: dfreproject.bib

---

# Summary
Coadded astronomical images generally consist of several, if not hundreds, of individual exposures. Each exposure is taken at 
a different time with minute differences in the positions of stars and other objects in the field due to the movement of the celestial bodies and changes/imperfections in the lense.
In order to combine individual exposures it is required to ensure that each exposure is properly aligned. 
Traditionally, this is done by reprojecting each exposure onto a common target grid defined in a World Coordinate System (WCS). 

In this package, we constructed functions that breakdown the coordinate transformations using Gnomonic projection to define 
the pixel-by-pixel shift from the source plane to the target plane. Additionally, we provide the requisite tools for interpolating the source image onto the target plane.
With a single function call, the user is able to calculate the complete reprojection of the source image onto the target plane.
This module follows the FITS and SIP formats layed out in the following papers: [@greisen_representations_2002], [@calabretta_representations_2002], and [@shupe_sip_2005].
We report a speedup of up to 40X when run on a GPU and 15X when run on a CPU compared to common alternatives.

# Statement of need

`dfreproject` is a Python package for astronomical image reprojection using `PyTorch` [@paszke_pytorch_2019] as the computational backbone.
This package was developed out of a need for a fast reprojection code that did not rely on pre-existing WCS calculations such as those found in `astropy` or `WCSLIB` [@astropy_collaboration_astropy_2013; @astropy_collaboration_astropy_2018; @astropy_collaboration_astropy_2022].
We do however use `astropy.wcs` to read the header information from the input fits files. 

Several packages already exist for calculating and applying the reprojection of a source image onto a target plane such as 
`reproject` [@robitaille_reproject_2020] or `astroalign` [@beroiz_astroalign_2020]. 
While these packages excel at easy-to-use, general-purpose astronomical image reprojection, they function solely on CPUs and therefore can serve as a computational bottleneck in data reduction pipelines.
It was with this in mind that we developed `dfreproject`.
`dfreproject`'s primary purpose is to reproject the observations taken by the new version of the Dragonfly Telescopic Array, Mothra. 
Mothra will contain 1000 individual lenses all simulataneously taking exposures with a cadence of a few minutes. 
Therefore, it is paramount to have a fast and accurate reprojection method.
By leveraging `PyTorch` for vectorization and parallelization via the GPU,
we are able to achieve a considerable speedup (up to nearly 40X) over standard methods.





# Methods
In order to reproject an image onto a new coordinate plane, we must perform three intermediate calculations. 
To do this, we use the target and source WCS. 
Before defining the steps, there are a few terms to define:
- SIP: Standard Imaging Polynomial. This convention allows us to represent non-linear geometric distortions as a simple polynomial.
The order and coefficients of this polynomial is stored in the header. The SIP is broken down into four individual polynomials, SIP\_A, SIP\_B, SIP\_INV\_A, and SIP\_INV\_B where
SIP\_A defines the polynomial applied to the x-coordinates, SIP\_B defines the polynomial applied to the y-coordinates, and SIP\_INV\_A and SIP\_INV\_B define the inverse operations.
For an in-depth discussion on SIP, please see @shupe_sip_2005.
- CD Matrix: Coordinate Description Matrix. This is  a 2X2 matrix that encodes the rotation, skew, and scaling of the image. 
The values are conveniently stored in the header. The CD matrix may also be constructed from the PC, Projection Coordinate, matrix multiplied by the CDELT values.

The steps are as follows:

1. For each pixel calculate the corresponding celestial coordinate using the target WCS

   1. Apply shift
   
   2. Apply SIP distortion

   3. Apply CD matrix
   
   4. Apply transformation to celestial coordinates using the Gnomonic projection


2. Calculate the position in the source grid for each celestial coordinate. This provides the offset for the next step.

   1. Apply the inverse transformation using the Gnomonic projection

   2. Apply inverse CD matrix

   3. Apply inverse SIP distortion

   4. Apply shift


3. Interpolate the source image onto the newly calculated grid 

In the final interpolation step we include local flux conservation by simultaneously projecting an identity tensor called the footprint.
The final reprojected frame is normalized by this footprint.

## Coordinate Transformation
In this section we describe the coordinate transformation using the Gnomonic projection. 
Please note that we include an additional shift of 1 pixel to handle the fact that Python is 0-based.
We will be using the following definitions for values: 

$x,y$ - pixel values

$\mathrm{crpix}_1, \mathrm{crpix}_2$ - center pixels as defined in WCS


$\mathrm{dec}_0, \mathrm{ra}_0$ - Central Declination and Right Ascension as defined in the WCS.


All trigonometric functions require the values be in radians.

### To celestial coordinates
For these calculations, we use the WCS information for the target plane.

$$u = x - (\mathrm{crpix}_1 - 1) $$
$$v = y - (\mathrm{crpix}_2 -1) $$

$$u = u - \mathrm{SIP\_A}(u, v)$$
$$v = v - \mathrm{SIP\_B}(u,v)$$

$$
\begin{bmatrix}
u' \\
v'
\end{bmatrix}
 = \mathrm{CD}
\begin{bmatrix}
u \\
v
\end{bmatrix}
$$
$$r = \sqrt{u'^2 + v'^2}$$
$$r_0 = \frac{180}{\pi}$$
$$\phi = atan^2(-u', v')$$
$$\theta = atan^2(r_0, r) $$
$$\mathrm{dec} = \sin^{-1}\Big( \sin(\theta)\sin(dec_0) + \cos(\theta)\cos(dec_0)\cos(\phi) \Big) $$
$$\mathrm{ra} = \mathrm{ra}_0 + \mathrm{atan}^2\Big( -\cos(\theta)\sin(\phi), \sin(\theta)\cos(dec_0)-\cos(\theta)\sin(dec_0)\cos(\phi) \Big) $$


### To source coordinates
For these calculations, we use the WCS information for the source plane.

$$\Delta \mathrm{ra} = \mathrm{ra} - \mathrm{ra}_0$$
$$\phi = \mathrm{atan}^2\Big(-\cos(\mathrm{dec})\sin(\Delta \mathrm{ra}), \sin(\mathrm{dec})\cos(\mathrm{dec}_0)-\cos(\mathrm{dec})\sin(\mathrm{dec}_0)\cos(\Delta \mathrm{ra}) \Big) $$
$$\theta = \sin^{-1}\Big( \sin(\mathrm{dec})\sin(\mathrm{dec}_0) + \cos(\mathrm{dec})\cos(\mathrm{dec}_0)\cos(\Delta \mathrm{ra}) \Big) $$
$$r = r_0\frac{\cos(\theta)}{\sin(\theta)}$$
$$u' = -r\sin(\phi) $$
$$v' = r\cos(\phi)$$



$$
\begin{bmatrix}
u \\
v
\end{bmatrix}
 = \mathrm{CD\_INV}
\begin{bmatrix}
u' \\
v'
\end{bmatrix}
$$

$$u = u - \mathrm{SIP\_INV\_A}(u, v)$$
$$v = v - \mathrm{SIP\_INV\_B}(u, v)$$

$$x = u + (\mathrm{crpix}_1 - 1)$$
$$y = v + (\mathrm{crpix}_2 - 1)$$

## Demo
For this demonstration we created two small (50x50) FITS files (see `demo/create_test_data::create_fits_file_tiny.py`) with a several pixel offset.
In \autoref{fig:demo} from left to right, we show the original image, the `dfreproject` solution, the `reproject` solution, and the relative error between the two.
For both solutions, we use a nearest-neighbor interpolation scheme.


![\label{fig:demo}](demo/comparison.png)


## Speed Comparison
In order to compare the execution times, we created a benchmarking script (that can be found in the demos/benchmarking directory under `benchmark_script.py`).
This test is run between `dfreproject` and `reproject`.
We benchmark the three interpolation schemes with and without SIP distortion for images sized 256x256, 512x512, 1024x1024, and 4000x6000\footnote{this matches the size of Dragonfly images}.
\autoref{fig:gpu-comparison} shows the results of this benchmarking when `dfreproject` is run using a GPU (NVIDIA GeForce RTX 4060).


![\label{fig:gpu-comparison}](demo/benchmarking/sip_comparison_line_gpu.png)


As evidenced by this figure, `dfreproject` has a significant speed advantage over `reproject` for larger images regardless of the type of interpolation scheme. 
The speedup is most pronounced in the case of where SIP distortions are included.

In \autoref{fig:cpu-comparison}, we display the same results except we used a CPU (Intel® Core™ i9-14900HX).


![\label{fig:cpu-comparison}](demo/benchmarking/sip_comparison_line_cpu.png)


Although the speedup on the CPU is not as impressive as on the GPU, it is still considerable.


# Acknowledgements
We would like to acknowledge the Dragonfly FRO. We would like to give a particularly warm thank you to Lisa Sloan for her project management skills.

We use the cmcrameri scientific color maps in our demos [@crameri_scientific_2023].

# References
