import jax.numpy as jnp


def cast_lasso_to_qp(X, y, penalization):
    n, d = X.shape
    p = y
    Q = jnp.eye(d) / 2
    A = jnp.concatenate([X, -X], axis=0)
    b = penalization * jnp.ones((2 * n,))
    return Q, p, A, b
