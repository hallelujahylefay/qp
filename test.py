import jax
import jax.numpy as jnp
from lasso import cast_lasso_to_qp
from newton import barr_method
from timeit import timeit
import matplotlib.pyplot as plt

shape = (2, 6)
mus = jnp.array([2, 15, 50, 100])
distrib = jax.random.uniform


OP_key = jax.random.PRNGKey(0)
keys = jax.random.split(OP_key, 200)
for mu in mus:

    @timeit
    @jax.vmap
    def wrapper_lasso_barr_method(key):
        penalization = 10
        X, y = distrib(key, shape=shape), distrib(key, shape=(shape[0],))
        Q, p, A, b = cast_lasso_to_qp(X, y, penalization=penalization)
        # feasible_v = penalization / 2 * jnp.ones((shape[0],)) / jnp.linalg.norm(A, jnp.inf)
        feasible_v = jnp.zeros((shape[0],))
        objective_function = jax.vmap(lambda v: v.T @ Q @ v + p.T @ v)
        _, vs = barr_method(Q, p, A, b, feasible_v, 1e-16, mu=mu)
        return vs, objective_function(vs)


    _, values = wrapper_lasso_barr_method(keys)
    for i in range(len(values)):
        trimed = jnp.trim_zeros(values[i])
        min = jnp.min(trimed)
        plt.semilogy(range(len(trimed)), trimed-min)
    plt.title(f"{len(keys)} lasso problems, semilog error, mu={mu}")
    plt.savefig(f"lasso_qp_{mu}", dpi=500)
    plt.close()