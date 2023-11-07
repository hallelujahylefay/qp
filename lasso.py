import jax.numpy as jnp


def cast_lasso_to_qp(X, y, penalization):
    n, d = X.shape
    p = jnp.zeros((d,))
    Q = jnp.eye(d) * penalization / 2
    A = jnp.concatenate([X, -X], axis=0)
    b = jnp.ones((2 * n,))
    return Q, p, A, b
