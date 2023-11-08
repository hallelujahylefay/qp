import jax
import jax.numpy as jnp
from lasso import cast_lasso_to_qp
from newton import barr_method
from timeit import timeit

shape = (2, 6)
mus = jnp.array([2, 15, 50, 100])
distrib = jax.random.uniform


@timeit
@jax.vmap
def wrapper_lasso_barr_method(key):
    penalization = 10
    X, y = distrib(key, shape=shape), distrib(key, shape=(shape[1],))
    Q, p, A, b = cast_lasso_to_qp(X, y, penalization=penalization)
    feasible_v = penalization / 2 * jnp.ones((shape[-1],)) / jnp.linalg.norm(A, jnp.inf)
    objective_function = jax.vmap(lambda v: v.T @ Q @ v + p.T @ v)
    _, vs = barr_method(Q, p, A, b, feasible_v, 1e-2, mu=10)
    return vs, objective_function(vs)


OP_key = jax.random.PRNGKey(0)
keys = jax.random.split(OP_key, 2)
with jax.disable_jit(True):
    wrapper_lasso_barr_method(keys)
