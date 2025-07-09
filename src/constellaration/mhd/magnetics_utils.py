import jaxtyping as jt
import numpy as np

from constellaration.geometry import radial_profile, surface_utils
from constellaration.mhd import vmec_utils


def vacuum_well(
    equilibrium: vmec_utils.VmecppWOut,
) -> float:
    r"""Computes a single number that summarizes the vacuum magnetic well, given by the
    formula.

    This function reproduces the `vacuum_well` function in Simsopt.

    The vacuum well is defined as:

    .. math::

        W = \frac{\partial_s V(s=0) - \partial_s V(s=1)}{\partial_s V(s=0)}

    where :math:`V` is the volume enclosed by the flux surface, and :math:`s` is the
    normalized toroidal flux. Positive values of :math:`W` are favorable for stability
    to interchange modes. This formula for :math:`W` is motivated by the fact that
    :math:`\tilde{W} = \frac{d^2 V}{d s^2} < 0` is favorable for stability. Integrating
    over :math:`\tilde{W}` from 0 to 1 and normalizing gives the above formula
    for :math:`W`.
    """

    # gmnc are the Fourier coefficients of the Jacobian in VMEC coordinates.
    # In VMEC, the radial derivative of the volume is (2pi)^2 * |g^{1/2}|.
    d_volume_d_s = 4 * np.pi * np.pi * np.abs(equilibrium.gmnc[:, 0])

    # Extrapolate linearly to the magnetic axis and the LCFS.
    d_volume_d_s_at_magnetic_axis = 1.5 * d_volume_d_s[0] - 0.5 * d_volume_d_s[1]
    d_volume_d_s_at_lcfs = 1.5 * d_volume_d_s[-1] - 0.5 * d_volume_d_s[-2]

    return (
        d_volume_d_s_at_magnetic_axis - d_volume_d_s_at_lcfs
    ) / d_volume_d_s_at_magnetic_axis


def magnetic_mirror_ratio(
    equilibrium: vmec_utils.VmecppWOut,
) -> radial_profile.InterpolatedRadialProfile:
    magnetic_field_strength = _magnetic_field_strength_nyquist_resolution(equilibrium)
    magnetic_field_strength_max = np.max(magnetic_field_strength, axis=(1, 2))
    magnetic_field_strength_min = np.min(magnetic_field_strength, axis=(1, 2))
    magnetic_mirror_ratio = (
        magnetic_field_strength_max - magnetic_field_strength_min
    ) / (magnetic_field_strength_max + magnetic_field_strength_min)
    return radial_profile.InterpolatedRadialProfile(
        rho=np.sqrt(equilibrium.normalized_toroidal_flux_full_grid_mesh),
        values=magnetic_mirror_ratio,
    )


def normalized_magnetic_gradient_scale_length(
    equilibrium: vmec_utils.VmecppWOut,
    theta_phi: jt.Float[np.ndarray, "n_poloidal_points n_toroidal_points 2"],
) -> jt.Float[np.ndarray, "n_poloidal_points n_toroidal_points"]:
    """Computes the magnetic gradient scale length.

    The quantity correlates with the coils-plasma distance for a given coil
    "complexity".

    This quantity is discussed in arXiv:2309.11342v1:

    The Magnetic Gradient Scale Length Explains Why Certain Plasmas
    Require Close External Magnetic Coils
    by John Kappel, Matt Landreman, and Dhairya Malhotra

    The base of this code was written by John Kappel and provided to us by Alan.
    Simsopt now also has an implementation available in the `vmec_compute_geometry`
    function.

    Args:
        equilibrium: The VMEC equilibrium.
        theta_phi: A grid of poloidal and toroidal angles at which to
            evaluate the magnetic gradient scale length.
    """

    ns = equilibrium.ns
    xm = equilibrium.xm
    xn = equilibrium.xn
    xm_nyq = equilibrium.xm_nyq
    xn_nyq = equilibrium.xn_nyq
    rmnc = equilibrium.rmnc
    zmns = equilibrium.zmns
    gmnc = equilibrium.gmnc
    bmnc = equilibrium.bmnc
    bsupumnc = equilibrium.bsupumnc
    bsupvmnc = equilibrium.bsupvmnc

    s_full = np.linspace(0, 1, ns)
    ds = s_full[2] - s_full[1]

    # Vectorize boundary derivative computations
    def get_d_x_d_s_at_the_boundary(x: np.ndarray, is_full_mesh: bool) -> np.ndarray:
        """Returns the derivative of x with respect to s at the plasma boundary.

        We need to extrapolate the derivative at the boundary because some
        quantities are not defined at the boundary.
        Coefficients for higher order approximations are taken from:
        https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf
        """
        if is_full_mesh:
            # 3rd order finite difference for full mesh
            return (11 / 6 * x[-1] - 3 * x[-2] + 1.5 * x[-3] - 1 / 3 * x[-4]) / ds
        else:
            # 2nd order finite difference for half mesh + boundary extrapolation
            d_x_d_s_n = (1.5 * x[-1] - 2.0 * x[-2] + 0.5 * x[-3]) / ds
            d_x_d_s_nm1 = (x[-1] - x[-3]) / (2 * ds)
            d_x_d_s_nm2 = (x[-2] - x[-4]) / (2 * ds)
            # Extrapolate to boundary
            return 7/4 * d_x_d_s_n - d_x_d_s_nm1 + 1/4 * d_x_d_s_nm2

    # Compute all boundary derivatives up front
    d_rmnc_d_s     = get_d_x_d_s_at_the_boundary(rmnc, True)
    d_zmns_d_s     = get_d_x_d_s_at_the_boundary(zmns, True)
    d_bmnc_d_s     = get_d_x_d_s_at_the_boundary(bmnc, False)
    d_bsupumnc_d_s = get_d_x_d_s_at_the_boundary(bsupumnc, False)
    d_bsupvmnc_d_s = get_d_x_d_s_at_the_boundary(bsupvmnc, False)

    # Vectorize boundary value computations
    def get_x_at_the_boundary(x: np.ndarray, is_full_mesh: bool) -> np.ndarray:
        """Returns the value of x at the plasma boundary."""
        if is_full_mesh:
            return x[-1]
        else:
            return 7 / 4 * x[-1] - x[-2] + 1 / 4 * x[-3]

    # Extract boundary values up front
    rmnc      = get_x_at_the_boundary(rmnc, True)
    zmns      = get_x_at_the_boundary(zmns, True)
    gmnc      = get_x_at_the_boundary(gmnc, False)
    bmnc      = get_x_at_the_boundary(bmnc, False)
    bsupumnc  = get_x_at_the_boundary(bsupumnc, False)
    bsupvmnc  = get_x_at_the_boundary(bsupvmnc, False)

    # Build array shapes for einsums:
    # All arrays above are shape (n_modes,)
    # theta2d/phi2d shape: (n_theta, n_phi)
    theta2d = theta_phi[..., 0]  # (n_theta, n_phi)
    phi2d = theta_phi[..., 1]

    # All 'modes' are shape (n_modes,)
    # Compute mode-angles for main and 'nyq' harmonics
    # We avoid newaxis/broadcast: use einsum
    # Precompute for main-mesh
    angle = np.einsum('i,jk->ijk', xm, theta2d) - np.einsum('i,jk->ijk', xn, phi2d)
    cos_of_angle = np.cos(angle)
    sin_of_angle = np.sin(angle)

    # Precompute for nyq-mesh arrays
    angle_nyq = np.einsum('i,jk->ijk', xm_nyq, theta2d) - np.einsum('i,jk->ijk', xn_nyq, phi2d)
    cos_of_nyq_angle = np.cos(angle_nyq)
    sin_of_nyq_angle = np.sin(angle_nyq)

    # Helper for fast spectral summation (mode contraction)
    def msum(values, arr):
        """Sum over mode index; avoid repeated broadcasting."""
        return np.einsum('m,mjk->jk', values, arr)

    # All operations below are now *matrix multiplications along mode axis*.
    # Precompute useful coefficients for basis multiplications.
    xm = xm.astype(float)
    xn = xn.astype(float)
    xm_nyq = xm_nyq.astype(float)
    xn_nyq = xn_nyq.astype(float)

    # Main mesh
    R             = msum(rmnc, cos_of_angle)
    d_R_d_theta   = msum(rmnc * xm, -sin_of_angle)
    d_R_d_phi     = msum(rmnc * -xn, -sin_of_angle)
    d2_R_d_theta2 = msum(rmnc * xm * xm, -cos_of_angle)
    d2_R_d_theta_d_phi = msum(rmnc * xm * -xn, -cos_of_angle)
    d2_R_d_phi2   = msum(rmnc * -xn * -xn, -cos_of_angle)

    d_Z_d_theta   = msum(zmns * xm, cos_of_angle)
    d_Z_d_phi     = msum(zmns * -xn, cos_of_angle)
    d2_Z_d_theta2 = msum(zmns * xm * xm, -sin_of_angle)
    d2_Z_d_theta_d_phi = msum(zmns * xm * -xn, -sin_of_angle)
    d2_Z_d_phi2   = msum(zmns * -xn * -xn, -sin_of_angle)

    d_R_d_s       = msum(d_rmnc_d_s, cos_of_angle)
    d_Z_d_s       = msum(d_zmns_d_s, sin_of_angle)
    d2_R_d_s_d_theta = msum(d_rmnc_d_s * xm, -sin_of_angle)
    d2_R_d_s_d_phi   = msum(d_rmnc_d_s * -xn, -sin_of_angle)
    d2_Z_d_s_d_theta = msum(d_zmns_d_s * xm, cos_of_angle)
    d2_Z_d_s_d_phi   = msum(d_zmns_d_s * -xn, cos_of_angle)

    # Nyq mesh
    B            = msum(bmnc, cos_of_nyq_angle)
    sqrt_g       = msum(gmnc, cos_of_nyq_angle)
    B_sup_theta  = msum(bsupumnc, cos_of_nyq_angle)
    B_sup_phi    = msum(bsupvmnc, cos_of_nyq_angle)
    d_B_sup_theta_d_theta = msum(bsupumnc * xm_nyq, -sin_of_nyq_angle)
    d_B_sup_phi_d_theta  = msum(bsupvmnc * xm_nyq, -sin_of_nyq_angle)
    d_B_sup_theta_d_phi  = msum(bsupumnc * -xn_nyq, -sin_of_nyq_angle)
    d_B_sup_phi_d_phi    = msum(bsupvmnc * -xn_nyq, -sin_of_nyq_angle)
    d_B_sup_theta_d_s    = msum(d_bsupumnc_d_s, cos_of_nyq_angle)
    d_B_sup_phi_d_s      = msum(d_bsupvmnc_d_s, cos_of_nyq_angle)

    # Compute cos/sin(phi) only once
    cos_of_phi = np.cos(phi2d)
    sin_of_phi = np.sin(phi2d)

    # Compute spatial derivatives (temporaries reused, broadcasting optimized)
    inv_sqrt_g = 1.0 / sqrt_g
    R_div_g = R * inv_sqrt_g

    grad_s__R   = -d_Z_d_theta * R_div_g
    grad_s__phi = (d_R_d_phi * d_Z_d_theta - d_R_d_theta * d_Z_d_phi) * inv_sqrt_g
    grad_s__Z   = d_R_d_theta * R_div_g

    grad_s__X = grad_s__R * cos_of_phi + grad_s__phi * -sin_of_phi
    grad_s__Y = grad_s__R * sin_of_phi + grad_s__phi * cos_of_phi

    grad_theta__R   = d_Z_d_s * R_div_g
    grad_theta__phi = (d_R_d_s * d_Z_d_phi - d_R_d_phi * d_Z_d_s) * inv_sqrt_g
    grad_theta__Z   = -d_R_d_s * R_div_g

    grad_theta__X = grad_theta__R * cos_of_phi + grad_theta__phi * -sin_of_phi
    grad_theta__Y = grad_theta__R * sin_of_phi + grad_theta__phi * cos_of_phi

    grad_phi__R   = np.zeros_like(sqrt_g)
    grad_phi__phi = 1 / R
    grad_phi__Z   = np.zeros_like(sqrt_g)
    grad_phi__X = grad_phi__R * cos_of_phi + grad_phi__phi * -sin_of_phi
    grad_phi__Y = grad_phi__R * sin_of_phi + grad_phi__phi * cos_of_phi

    # Compute all B derivatives in X, Y, Z efficiently, reusing intermediates
    # Macro: d_B_X_s, d_B_Y_s, d_B_Z_s, etc.
    # Avoid deep parenthesis, expand = single expressions and let numpy broadcast and fuse

    d_B_X_d_s = (
        d_B_sup_theta_d_s * d_R_d_theta * cos_of_phi +
        B_sup_theta * d2_R_d_s_d_theta * cos_of_phi +
        d_B_sup_phi_d_s * d_R_d_phi * cos_of_phi +
        B_sup_phi * d2_R_d_s_d_phi * cos_of_phi -
        d_B_sup_phi_d_s * R * sin_of_phi -
        B_sup_phi * d_R_d_s * sin_of_phi
    )
    d_B_X_d_theta = (
        d_B_sup_theta_d_theta * d_R_d_theta * cos_of_phi +
        B_sup_theta * d2_R_d_theta2 * cos_of_phi +
        d_B_sup_phi_d_theta * d_R_d_phi * cos_of_phi +
        B_sup_phi * d2_R_d_theta_d_phi * cos_of_phi -
        d_B_sup_phi_d_theta * R * sin_of_phi -
        B_sup_phi * d_R_d_theta * sin_of_phi
    )
    d_B_X_d_phi = (
        d_B_sup_theta_d_phi * d_R_d_theta * cos_of_phi +
        B_sup_theta * d2_R_d_theta_d_phi * cos_of_phi -
        B_sup_theta * d_R_d_theta * sin_of_phi +
        d_B_sup_phi_d_phi * d_R_d_phi * cos_of_phi +
        B_sup_phi * d2_R_d_phi2 * cos_of_phi -
        B_sup_phi * d_R_d_phi * sin_of_phi -
        d_B_sup_phi_d_phi * R * sin_of_phi -
        B_sup_phi * d_R_d_phi * sin_of_phi -
        B_sup_phi * R * cos_of_phi
    )

    d_B_Y_d_s = (
        d_B_sup_theta_d_s * d_R_d_theta * sin_of_phi +
        B_sup_theta * d2_R_d_s_d_theta * sin_of_phi +
        d_B_sup_phi_d_s * d_R_d_phi * sin_of_phi +
        B_sup_phi * d2_R_d_s_d_phi * sin_of_phi +
        d_B_sup_phi_d_s * R * cos_of_phi +
        B_sup_phi * d_R_d_s * cos_of_phi
    )
    d_B_Y_d_theta = (
        d_B_sup_theta_d_theta * d_R_d_theta * sin_of_phi +
        B_sup_theta * d2_R_d_theta2 * sin_of_phi +
        d_B_sup_phi_d_theta * d_R_d_phi * sin_of_phi +
        B_sup_phi * d2_R_d_theta_d_phi * sin_of_phi +
        d_B_sup_phi_d_theta * R * cos_of_phi +
        B_sup_phi * d_R_d_theta * cos_of_phi
    )
    d_B_Y_d_phi = (
        d_B_sup_theta_d_phi * d_R_d_theta * sin_of_phi +
        B_sup_theta * d2_R_d_theta_d_phi * sin_of_phi +
        B_sup_theta * d_R_d_theta * cos_of_phi +
        d_B_sup_phi_d_phi * d_R_d_phi * sin_of_phi +
        B_sup_phi * d2_R_d_phi2 * sin_of_phi +
        B_sup_phi * d_R_d_phi * cos_of_phi +
        d_B_sup_phi_d_phi * R * cos_of_phi +
        B_sup_phi * d_R_d_phi * cos_of_phi -
        B_sup_phi * R * sin_of_phi
    )

    d_B_Z_d_s = (
        d_B_sup_theta_d_s * d_Z_d_theta +
        B_sup_theta * d2_Z_d_s_d_theta +
        d_B_sup_phi_d_s * d_Z_d_phi +
        B_sup_phi * d2_Z_d_s_d_phi
    )
    d_B_Z_d_theta = (
        d_B_sup_theta_d_theta * d_Z_d_theta +
        B_sup_theta * d2_Z_d_theta2 +
        d_B_sup_phi_d_theta * d_Z_d_phi +
        B_sup_phi * d2_Z_d_theta_d_phi
    )
    d_B_Z_d_phi = (
        d_B_sup_theta_d_phi * d_Z_d_theta +
        B_sup_theta * d2_Z_d_theta_d_phi +
        d_B_sup_phi_d_phi * d_Z_d_phi +
        B_sup_phi * d2_Z_d_phi2
    )

    # Now build full gradient matrix and contraction
    # We avoid a huge number of repeated temporaries by using in-place adds to accum array
    grad_B_double_dot_grad_B = np.zeros_like(B)
    for grad_B in (
        d_B_X_d_s * grad_s__X + d_B_X_d_theta * grad_theta__X + d_B_X_d_phi * grad_phi__X,
        d_B_X_d_s * grad_s__Y + d_B_X_d_theta * grad_theta__Y + d_B_X_d_phi * grad_phi__Y,
        d_B_X_d_s * grad_s__Z + d_B_X_d_theta * grad_theta__Z + d_B_X_d_phi * grad_phi__Z,
        d_B_Y_d_s * grad_s__X + d_B_Y_d_theta * grad_theta__X + d_B_Y_d_phi * grad_phi__X,
        d_B_Y_d_s * grad_s__Y + d_B_Y_d_theta * grad_theta__Y + d_B_Y_d_phi * grad_phi__Y,
        d_B_Y_d_s * grad_s__Z + d_B_Y_d_theta * grad_theta__Z + d_B_Y_d_phi * grad_phi__Z,
        d_B_Z_d_s * grad_s__X + d_B_Z_d_theta * grad_theta__X + d_B_Z_d_phi * grad_phi__X,
        d_B_Z_d_s * grad_s__Y + d_B_Z_d_theta * grad_theta__Y + d_B_Z_d_phi * grad_phi__Y,
        d_B_Z_d_s * grad_s__Z + d_B_Z_d_theta * grad_theta__Z + d_B_Z_d_phi * grad_phi__Z
    ):
        grad_B_double_dot_grad_B += grad_B**2

    magnetic_gradient_scale_length = B * np.sqrt(2 / grad_B_double_dot_grad_B)
    return magnetic_gradient_scale_length / equilibrium.Aminor_p


def _magnetic_field_strength_nyquist_resolution(
    equilibrium: vmec_utils.VmecppWOut,
) -> jt.Float[np.ndarray, "n_flux_surfaces n_poloidal_points n_toroidal_points 3"]:
    (
        n_poloidal_points,
        n_toroidal_points,
    ) = surface_utils.n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
        n_poloidal_modes=equilibrium.mpol,
        max_toroidal_mode=equilibrium.ntor,
    )
    phi_upper_bound = (
        2 * np.pi / equilibrium.n_field_periods / (1 + int(not equilibrium.lasym))
    )
    s_theta_phi = surface_utils.make_s_theta_phi_grid(
        n_radial_points=equilibrium.ns,
        n_poloidal_points=n_poloidal_points,
        n_toroidal_points=n_toroidal_points,
        phi_upper_bound=phi_upper_bound,
        include_endpoints=True,
    )
    return vmec_utils.magnetic_field_magnitude(
        equilibrium=equilibrium,
        s_theta_phi=s_theta_phi,
    )
