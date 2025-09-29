from jax import numpy as jnp
from numpyro import deterministic, sample
from numpyro import distributions as dist


# Simplest model with Dirichlet prior on mz_intensities
def ms_model(mzs, intensities=None, n_components=5):
    N_spec, N_mz = intensities.shape if intensities is not None else (1, mzs.shape[0])
    tic = intensities.sum(axis=1) if intensities is not None else None
    rel_tic = (tic / tic.min()) if tic is not None else 1.0
    intensities_norm = intensities / tic[:, None] if tic is not None else None

    component_intensities = sample("component_intensities", dist.Dirichlet(0.25*jnp.ones(N_mz)).expand([n_components]))
    component_weights = sample("component_weights", dist.Dirichlet(0.5*jnp.ones(n_components)).expand([N_spec]))
    mz_intensities = deterministic("mz_intensities", jnp.matmul(component_weights, component_intensities))
    
    deviation = deterministic("deviation", (intensities_norm - mz_intensities) * 250.0 * rel_tic[:, None]) if intensities is not None else None
    
    if intensities is not None:
        sample("intensity_obs_norm", dist.Normal(0, 1.0), obs=deviation)

    return locals()
