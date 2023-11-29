# script to apply MLE to synthetic GBM data
import jax.numpy as jnp
import jax
from jax import jit, grad, vmap
import matplotlib.pyplot as plt


def get_data(params, delta_t, n, key):
    mu, sigma = params[0], params[1]
    z = jax.random.normal(key, shape=(n,))
    x = 1.0
    ret = [x]
    for i in range(1, n):
        inc = (mu-0.5*(sigma**2))*delta_t + sigma*jnp.sqrt(delta_t)*z[i]
        x *= jnp.exp(inc)
        ret.append(x)
    return jnp.array(ret)


# define loss function for a single data point
def loss_fn_sample(params, r, delta_t):
    mu, sigma = params[0], params[1]
    t1 = jnp.log(sigma)
    t2 = (r-(mu-0.5*(sigma**2))*delta_t)**2 / (2*(sigma**2)*delta_t)
    return t1+t2


# vectorize the loss function
loss_fn_samples = vmap(loss_fn_sample, in_axes=(None, 0, None))


@jax.jit
# compute the mean loss
def loss_fn(params, rs, delta_t):
    return jnp.mean(loss_fn_samples(params, rs, delta_t))


grad_fn = grad(loss_fn)


# define a newton step
@jax.jit
def newton_step(rs, params, delta_t):
    grad = grad_fn(params, rs, delta_t)
    hess = jax.hessian(loss_fn)(params, rs, delta_t)
    params -= jnp.linalg.inv(hess)@grad
    loss = loss_fn(params, rs, delta_t)
    return params, loss


# True Parameters
true_params = jnp.array([0.1, 0.3])
delta_t = 0.1

# Generating the GBM data
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)


ns = [100, 500, 1000, 5000, 10000]
errors = []

for n in ns:
    data = get_data(true_params, delta_t, n, key=subkey)
    # compute the log returns
    log_returns = jnp.log(data[1:]/data[:-1])

    # initialize the parameters
    params = jnp.array([0.2, 0.2])
    num_steps = 50

    # run the newton steps
    for i in range(num_steps):
        params, loss = newton_step(log_returns, params, delta_t)

    # compute estimation error
    error = jnp.abs(params-true_params)
    errors.append(error)

errors = jnp.array(errors)
# plot the results
plt.plot(ns, errors[:, 0], '-o', label='$|\hat{\mu}-\mu|$')
plt.plot(ns, errors[:, 1], '-o', label='$|\hat{\sigma}-\sigma|$')
plt.xlabel('Num. of Data Points')
plt.ylabel('Estimation Error')
plt.title('MLE Error vs. Num. of Data Points')
plt.xscale('log')
plt.legend()
plt.savefig('gbm_mle.pdf', dpi=300)
plt.close()
