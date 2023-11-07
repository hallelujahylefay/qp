from functools import partial
import jax
import jax.numpy as jnp
from lasso import cast_lasso_to_qp
from newton import barr_method

shape = (4, 5)
distrib = jax.random.uniform


@jax.vmap
def test(key):
    X, y = distrib(key, shape=shape), distrib(key, shape=(shape[0],))
    Q, p, A, b = cast_lasso_to_qp(X, y, penalization=10)
    return barr_method(Q, p, A, b, jnp.zeros((shape[-1],)), 1e-1)


OP_key = jax.random.PRNGKey(0)
keys = jax.random.split(OP_key, 2)

test(keys)
# shape = jax.lax.sort(jax.random.randint(key, (N, 2), 1, 100))
