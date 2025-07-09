import jaxtyping as jt
import numpy as np
import pydantic

NpOrJaxArray = np.ndarray | jt.Array


class FluxPowerSeriesProfile(pydantic.BaseModel):
    r"""A radial profile whose values are defined by a power series
    in the normalized toroidal flux:

    .. math::
        f(s) = \sum_{n=0}^{N} a_n s^n

    where :math:`s` is the normalized toroidal flux,
    and :math:`a_n` are the coefficients of the power series.
    """

    coefficients: list[float]


def evaluate_derivative(
    profile: FluxPowerSeriesProfile,
) -> FluxPowerSeriesProfile:
    return _evaluate_nth_derivative(profile, n=1)


def evaluate_at_normalized_effective_radius(
    profile: FluxPowerSeriesProfile,
    normalized_effective_radius: jt.Float[NpOrJaxArray, " n_points"],
) -> jt.Float[NpOrJaxArray, " n_points"]:
    return evaluate_at_normalized_toroidal_flux(profile, normalized_effective_radius**2)


def evaluate_at_normalized_toroidal_flux(
    profile: FluxPowerSeriesProfile,
    normalized_toroidal_flux: jt.Float[NpOrJaxArray, " n_points"],
) -> jt.Float[NpOrJaxArray, " n_points"]:
    # Vectorized power series evaluation for improved performance.
    coefs = np.asarray(profile.coefficients)
    # shape: (n_coefs, n_points)
    powers = np.power.outer(normalized_toroidal_flux, np.arange(coefs.size)).T
    # coefs shape: (n_coefs,)
    # powers shape: (n_coefs, n_points)
    # Weighted sum over all coefficients, shape: (n_points,)
    return np.dot(coefs, powers)


def _evaluate_nth_derivative(
    profile: FluxPowerSeriesProfile, n: int
) -> FluxPowerSeriesProfile:
    coefficients = profile.coefficients
    for _ in range(n):
        coefficients = [a_n * i for i, a_n in enumerate(coefficients) if i > 0]
    return FluxPowerSeriesProfile(coefficients=coefficients)
