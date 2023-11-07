import jax
import jax.numpy as jnp

beta = 0.99
alpha = 0.01


def centering_step(Q, p, A, b, t, v0, eps):
    def objective(v):
        return v.T @ Q @ v + p.T @ v

    def centering_objective(t, v):
        return t * objective(v) - jnp.sum(jnp.log(b - A @ v))

    def cond(args):
        lambda_square, _, _ = args
        return lambda_square / 2 > eps

    def iter_newton(inps):
        _, v, t = inps
        _centering_objective = lambda _v: centering_objective(t, _v)

        jac, hessian = jax.jacfwd(_centering_objective)(v), jax.hessian(_centering_objective)(v)
        Deltax_nt = - jnp.linalg.inv(hessian) @ jac
        lambda_square = - jac @ Deltax_nt
        t_star = backtracking_line_search(v, Deltax_nt, jac)
        v += t_star * Deltax_nt
        return lambda_square, v, t

    def backtracking_line_search(v, Deltax_nt, jac):
        def cond(t_p):
            return centering_objective(t, v + t_p * Deltax_nt) >= centering_objective(t,
                                                                                      v) + alpha * t_p * jac.T @ Deltax_nt

        def iter_backtracking(t_p):
            return t_p * beta

        t_star = jax.lax.while_loop(cond, iter_backtracking, 1.0)
        return t_star

    _, v, t = jax.lax.while_loop(cond, iter_newton, (3 * eps, v0, t))
    return v


def barr_method(Q, p, A, b, v0, eps, mu=10):
    def iter_barrier(inps):
        v, t = inps
        v_star = centering_step(Q, p, A, b, t, v, eps)
        t *= mu
        return v_star, t

    def cond(args):
        _, t = args
        return 1 / t > eps

    v_optimal, _ = jax.lax.while_loop(cond, iter_barrier, (v0, 1))
    return v_optimal
