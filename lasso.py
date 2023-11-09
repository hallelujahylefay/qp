import jax.numpy as jnp
import jax
from newton import barr_method, qp


@jax.jit
def from_dual_to_primal(v, X, y):
    return jnp.linalg.lstsq(X, y + v)[0]


@jax.jit
def lasso_objective(w, X, y, penalization):
    return 1 / 2 * jnp.linalg.norm(X @ w - y) ** 2 + penalization * jnp.linalg.norm(w, ord=1)


@jax.jit
def cast_lasso_to_qp(X, y, penalization):
    n, d = X.shape
    Q = jnp.eye(n) / 2
    p = y
    A = jnp.vstack([X.T, -X.T])
    b = jnp.full(2 * d, penalization)
    return Q, p, A, b


@jax.jit
def solve_lasso(X, y, penalization, mu, eps=1e-16):
    shape = X.shape
    Q, p, A, b = cast_lasso_to_qp(X, y, penalization)
    feasible_v = jnp.zeros((shape[0],))

    n_iter, _, duals = barr_method(Q, p, A, b, feasible_v, eps, mu)

    dual_objective_vmap = jax.vmap(qp, in_axes=(0, None, None))
    lasso_objective_vmap = jax.vmap(lasso_objective, in_axes=(0, None, None, None))
    from_dual_to_primal_vmap = jax.vmap(from_dual_to_primal, in_axes=(0, None, None))

    primals = from_dual_to_primal_vmap(duals, X, y)
    values_for_dual_objective = dual_objective_vmap(duals, Q, p)
    lasso_objective_values = lasso_objective_vmap(primals, X, y, penalization)

    return n_iter, duals, primals, values_for_dual_objective, lasso_objective_values
