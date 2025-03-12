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
  - name: Steven Janssens
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
Complete astronomical images generally consist of several, if not hundreds, of individual exposures. Each exposure is taken at 
a different time with minute differences in the positions of stars and other objects in the field due to the movement of the celestial bodies and changes/imperfections in the lense.
In order to combine, or stack, individual exposures it is required to ensure that each exposure is properly aligned. 
Traditionally, this is done by reprojecting each exposure onto a common target grid defined in World Coordinate System (WCS).  
In this package, we constructed functions that breakdown the coordinate transformations using Gnomonic projection to define 
the pixel-by-pixel shift from the source plane to the target plane. Additionally, we provide the requisite tools for interpolating the source image onto the target plane.
With a single function call, the user is able to calculate the complete reprojection of the source image onto the target plane.
This module follows the FITS and SIP formats layed out in the following papers: [@greisen_representations_2002], [@calabretta_representations_2002], and [@shupe_sip_2005].


# Statement of need

`dfreproject` is a Python package developed using `PyTorch` [@paszke_pytorch_2019] as the computational backbone for astronomical image reprojection.
This package was developed out of a need for a fast reprojection code that did not rely on pre-existing WCS calculations (such as those found in `astropy` or `WCSLIB`; [@astropy_collaboration_astropy_2013; @astropy_collaboration_astropy_2018; @astropy_collaboration_astropy_2022]).
We do however use `astropy.wcs` to read the header information from the input fits files. 

Although several packages already exist for calculating and applying the reprojection of a source image onto a target plane such as 
`reproject` [@robitaille_reproject_2020] or `astroalign` [@beroiz_astroalign_2020]. 
While these packages excel at easy-to-use, general-purpose astronomical image reprojection, they function solely on CPUs and therefore can serve as a computational bottleneck in data reduction pipelines.
It was with this in mind that we developed `dfreproject`. By leveraging `PyTorch` for vectorization and parallelization via the GPU,
we are able to achieve a considerable speedup (up to nearly 40X) over standard methods.





# Methods
In order to reproject an image onto a new coordinate plane, we must perform three intermediate calculations. 
To do this, we use target WCS and source WCS. The steps are as follows:
1. For each pixel calculate the corresponding celestial coordinate using the target WCS
   1. Apply shift
   2. Apply SIP distortion
   3. Apply CD matrix
   4. Apply transformation to celestial coordinates uing the Gnomonic Projection
2. Calculate the position in the source grid for each celestial coordinate. This provides the offset for the next step.
   1. Apply the inverse transformation using the Gnomonic projecction
   2. Apply inverse CD matrix
   3. Apply inverse SIP distortion
   4. Apply shift
3. Interpolate the source image onto the newly calculated grid 

In the final interpolation step we include local flux conservation by simulatenously projecting an identity tensor called the footprint.
The final reprojected frame is normalized by this footprint.

## Coordinate Transformation
In this section we describe the coordinate transformation using the Gnomonic projection. Please note that we include an additional shift of 1 pixel to handle the fact that python is 0-based.
We will be using the following definintions for values: 

$x,y$ - pixel values

$crpix_1, crpix_2$ - center pixels as defined in WCS

$CD$ - CD matrix as defined in WCS. This can also be constructed from  the PC matrix using the following relation $CD = PC * CDELT$ where $CDELT=[cdelt_1, cdelt_2]$.

$SIP\_A, SIP\_B$ - SIP polynomials A and B 

$SIP\_INV\_A, SIP\_INV\_B$ - Inverse SIP polynomials A and B

$dec_0, ra_0$ - Central Declination and Right Ascension as defined in the WCS.


All trignometric functions require the values be in radians.

### To celestial coordinates
For these calculations, we use the WCS information for the target plane.

$$u = x - (crpix_1 - 1) $$
$$v = y - (crpix_2 -1) $$

$$u = u - SIP\_A(u, v)$$
$$v = v - SIP\_B(u,v)$$

$$
\begin{bmatrix}
u' \\
v'
\end{bmatrix}
 = CD 
\begin{bmatrix}
u \\
v
\end{bmatrix}
$$
$$r = \sqrt{u'^2 + v'^2}$$
$$r_0 = \frac{180}{\pi}$$
$$\phi = atan^2(-u', v')$$
$$\theta = atan^2(r_0, r) $$
$$dec = sin^{-1}\Big( sin(\theta)sin(dec_0) + cos(\theta)cos(dec_0)cos(\phi) \Big) $$
$$ra = ra_0 + atan^2\Big( -cos(\theta)sin(\phi), sin(\theta)cos(dec_0)-cos(\theta)sin(dec_0)cos(\phi) \Big) $$


### To source coordinates
For these calcuations, we use the WCS ionformation for the source plane.

$$\Delta ra = ra - ra_0$$
$$\phi = atan^2\Big(-cos(dec)sin(\Delta ra), sin(dec)cos(dec_0)-cos(dec)sin(dec_0)cos(\Delta ra) \Big) $$
$$\theta = sin^{-1}\Big( sin(dec)sin(dec_0) + cos(dec)cos(dec_0)cos(\Delta ra) \Big) $$
$$r = r_0\frac{cos(\theta)}{sin(\theta)}$$
$$u' = -rsin(\phi) $$
$$v' = rcos(\phi)$$



$$
\begin{bmatrix}
u \\
v
\end{bmatrix}
 = CD\_INV
\begin{bmatrix}
u' \\
v'
\end{bmatrix}
$$

$$u = u - SIP\_INV\_A(u, v)$$
$$v = v - SIP\_INV\_B(u, v)$$

$$x = u + (crpix_1 - 1)$$
$$y = v + (crpix_2 - 1)$$

## Demo
For this demonstration we created two small (50x50) fits files (see `demo/create_test_data::create_fits_file_tiny`) with a several pixel offset.
In figure \autoref{fig:demo}, you can see the original image, the `dfreproject` solution, the `reproject` solution, and the relative error between the two.
For both solutions, we use a nearest-neighbor interpolation scheme.


![\label{fig:demo}](demo/comparison.png)


## Speed Comparison
In order to compare the execution times, we created a benchmarking script (that can be found in the demos/benchmarking directory under `benchmark_script.py`).
We benchmark the three interpolation schemes with and without SIP distortion for images sized 256x256, 512x512, 1024x1024, and 4000x600.
Figure \autoref{fig:gpu-comparison} shows the results of this benchmarking when `dfreproject` is run using a GPU (NVIDIA GeForce RTX 4060).


![\label{fig:gpu-comparison}](demo/benchmarking/sip_comparison_line_gpu.png)


As evidenced by this figure, `dfreproject` has a significant speed advantage over `reproject` for larger images regardless of the type of interpolation scheme. 
The speedup is most pronounced in the case of where SIP distortions are included.

In figure \autoref{fig:cpu-comparison}, we display the same results except we used a CPU (Intel® Core™ i9-14900HX).


![\label{fig:cpu-comparison}](demo/benchmarking/sip_comparison_line_cpu.png)


Although the speedup on the CPU is not as impressive as on the GPU, it is still considerable.


# Acknowledgements
We would like to acknowledge the Dragonfly FRO. We would like to give a particularly warm thank you to Lisa Sloan for her project management skills.

We use the cmcrameri scientific color maps in our demos [@crameri_scientific_2023].

# References
