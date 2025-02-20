import torch
import math


def safe_tangent_projection(ra_rad, dec_rad, ra_0, dec_0):
    """
    Safe tangent plane (gnomonic) projection

    Parameters:
    - ra_rad: Right Ascension in radians (tensor)
    - dec_rad: Declination in radians (tensor)
    - ra_0: Reference Right Ascension in degrees
    - dec_0: Reference Declination in degrees

    Returns:
    - x, y coordinates in the tangent plane
    """
    # Convert reference coordinates to radians
    ra_0_rad = torch.deg2rad(torch.tensor(ra_0, dtype=ra_rad.dtype))
    dec_0_rad = torch.deg2rad(torch.tensor(dec_0, dtype=dec_rad.dtype))

    # Compute differences
    d_ra = ra_rad - ra_0_rad

    # Ensure d_ra is within [-pi, pi]
    d_ra = torch.where(d_ra > math.pi, d_ra - 2 * math.pi,
                       torch.where(d_ra < -math.pi, d_ra + 2 * math.pi, d_ra))

    # Trigonometric terms
    cos_dec = torch.cos(dec_rad)
    sin_dec = torch.sin(dec_rad)
    cos_dec_0 = torch.cos(dec_0_rad)
    sin_dec_0 = torch.sin(dec_0_rad)
    cos_d_ra = torch.cos(d_ra)
    sin_d_ra = torch.sin(d_ra)

    # Compute projection terms
    cos_c = sin_dec_0 * sin_dec + cos_dec_0 * cos_dec * cos_d_ra

    # Prevent division by zero or near-zero
    eps = 1e-10
    cos_c = torch.clamp(cos_c, min=-1 + eps, max=1 - eps)

    # Projection coordinates
    x = cos_dec * sin_d_ra / cos_c
    y = (sin_dec * cos_dec_0 - cos_dec * sin_dec_0 * cos_d_ra) / cos_c

    return x, y


def debug_projection(ra_coords, dec_coords, ra_0, dec_0):
    """
    Debug the projection with comprehensive output
    """
    x, y = safe_tangent_projection(ra_coords, dec_coords, ra_0, dec_0)

    print(f"Reference Point: RA_0 = {ra_0}, DEC_0 = {dec_0}")
    print(f"Input RA range: {torch.rad2deg(ra_coords.min()):.5f} to {torch.rad2deg(ra_coords.max()):.5f}")
    print(f"Input DEC range: {torch.rad2deg(dec_coords.min()):.5f} to {torch.rad2deg(dec_coords.max()):.5f}")
    print(f"x range: {x.min():.5f} to {x.max():.5f}")
    print(f"y range: {y.min():.5f} to {y.max():.5f}")

    return x, y


# Example usage
ra_coords = torch.deg2rad(torch.tensor([84.12094, 84.22094, 84.32094]))
dec_coords = torch.deg2rad(torch.tensor([-4.3705, -4.2705, -4.1705]))
debug_projection(ra_coords, dec_coords, 84.12094, -4.3705)