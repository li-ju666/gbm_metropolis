# script to apply Bayesian inference to synthetic GBM data
import jax.numpy as jnp
import jax
from jax import jit, grad, vmap
from jax.scipy.stats import norm, gamma
import seaborn as sns
import matplotlib.pyplot as plt


def get_data(params, delta_t, n, key):
    mu, sigma = params
    z = jax.random.normal(key, shape=(n,))
    x = 1.0
    ret = [x]
    for i in range(1, n):
        inc = (mu-0.5*(sigma**2))*delta_t + sigma*jnp.sqrt(delta_t)*z[i]
        x *= jnp.exp(inc)
        ret.append(x)
    return jnp.array(ret)


# define the prior logpdfs
def prior_logpdf(params):
    mu, sigma = params
    mu_logpdf = norm.logpdf(mu, 0.0, 1.0)
    sigma_logpdf = gamma.logpdf(sigma, 1.0)
    return mu_logpdf + sigma_logpdf


# define the loglikelihood
def loglikelihood_point(params, log_return, delta_t):
    mu, sigma = params
    t1 = -jnp.log(sigma)
    t2 = -(log_return-(mu-0.5*(sigma**2))*delta_t)**2 / (2*(sigma**2)*delta_t)
    return t1+t2


# vectorize the loglikelihood function
loglikelihood_points = vmap(loglikelihood_point, in_axes=(None, 0, None))


# define the sum of loglikelihoods
def loglikelihood(params, log_returns, delta_t):
    ret = jnp.sum(loglikelihood_points(params, log_returns, delta_t))
    return ret


# define the posterior logpdf
def posterior_logpdf(params, log_returns, delta_t):
    ret = prior_logpdf(params) + \
        loglikelihood(params, log_returns, delta_t)
    return ret


def compute_probability(new_params, params, log_returns, delta_t):
    log_alpha = posterior_logpdf(new_params, log_returns, delta_t) - \
        posterior_logpdf(params, log_returns, delta_t)
    return jnp.exp(log_alpha)


@jit
def mcmc_step(params, log_returns, delta_t, key):
    muk, sigmak, uk = jax.random.split(key, 3)
    # propose a step
    new_mu = jax.random.normal(muk)*0.1 + params[0]
    # new_sigma = jax.random.gamma(sigmak, 1.0)
    new_sigma = params[1] * jnp.exp(jax.random.normal(sigmak)*0.01)
    new_params = jnp.array([new_mu, new_sigma])

    # compute the acceptance probability
    alpha = compute_probability(
        new_params, params, log_returns, delta_t)

    # accept or reject
    u = jax.random.uniform(uk)
    return jax.lax.cond(
        u < alpha,
        lambda x: (new_params, True),
        lambda x: (params, False), None)


def mcmc(params, log_returns, delta_t, key, num_steps):
    samples = [params]
    keys = jax.random.split(key, num_steps)
    for idx, key in enumerate(keys):
        params, accepted = mcmc_step(
            params, log_returns, delta_t, key)
        if accepted:
            samples.append(params)

        # if idx % num_steps//10 == 0:
        #     print(jnp.mean(jnp.array(samples), axis=0))
    return jnp.array(samples)


def main():
    # True Parameters
    true_params = jnp.array([0.1, 0.3])
    # T = 100.0
    delta_t = 0.1
    ns = [100, 500, 1000, 5000, 10000]

    mcmc_samples = {}

    for n in ns:
        # Generating the GBM data
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)

        data = get_data(true_params, delta_t, n, key=subkey)

        # compute the log returns
        log_returns = jnp.log(data[1:]/data[:-1])

        num_steps = int(1e4)
        # start from expectation of prior
        start = jnp.array([0., 1.])

        key, subkey = jax.random.split(key)
        samples = mcmc(start, log_returns,
                       delta_t, subkey, num_steps)

        mcmc_samples[n] = samples[samples.shape[0]//2:]
        print(jnp.mean(mcmc_samples[n], axis=0))

    # plot the kde from samples for mu and sigma separately
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    for n in ns:
        label = f'N={n}'
        sns.kdeplot(
            mcmc_samples[n][:, 0], ax=ax[0], label=label)
        sns.kdeplot(
            mcmc_samples[n][:, 1], ax=ax[1], label=label)

    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('$\mu$')
    ax[1].set_xlabel('$\sigma$')
    ax[0].set_ylabel('Density')
    ax[1].set_ylabel('Density')

    title = 'Posterior Densities for $\mu$ and $\sigma$'
    fig.suptitle(title)

    plt.savefig('gbm_bayes.pdf', dpi=300)


if __name__ == "__main__":
    main()
