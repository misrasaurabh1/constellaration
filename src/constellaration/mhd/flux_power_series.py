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
    # Fast path: n=1 is common, use a tight loop
    coefficients = profile.coefficients
    if not coefficients or len(coefficients) < 2:
        return FluxPowerSeriesProfile(coefficients=[])
    deriv_coeff = [coefficients[i] * i for i in range(1, len(coefficients))]
    return FluxPowerSeriesProfile(coefficients=deriv_coeff)


def evaluate_at_normalized_effective_radius(
    profile: FluxPowerSeriesProfile,
    normalized_effective_radius: jt.Float[NpOrJaxArray, " n_points"],
) -> jt.Float[NpOrJaxArray, " n_points"]:
    return evaluate_at_normalized_toroidal_flux(profile, normalized_effective_radius**2)


def evaluate_at_normalized_toroidal_flux(
    profile: FluxPowerSeriesProfile,
    normalized_toroidal_flux: jt.Float[NpOrJaxArray, " n_points"],
) -> jt.Float[NpOrJaxArray, " n_points"]:
    return np.sum(
        [
            a_n * normalized_toroidal_flux**n
            for n, a_n in enumerate(profile.coefficients)
        ],
        axis=0,
    )


def _evaluate_nth_derivative(
    profile: FluxPowerSeriesProfile, n: int
) -> FluxPowerSeriesProfile:
    coefficients = profile.coefficients
    for _ in range(n):
        coefficients = [a_n * i for i, a_n in enumerate(coefficients) if i > 0]
    return FluxPowerSeriesProfile(coefficients=coefficients)
