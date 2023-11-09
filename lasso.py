import jax.numpy as jnp


def cast_lasso_to_qp(X, y, penalization):
    n, d = X.shape
    Q = jnp.eye(n) / 2
    p = y
    A = jnp.concatenate([X.T, -X.T])
    b = jnp.full(2 * d, penalization)
    return Q, p, A, b
