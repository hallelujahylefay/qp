from qp import qp
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


@jax.jit
def centering_step(Q, p, A, b, t, v0, eps, max_iter=10):
    """
    Centering step of the barrier method for quadratic programs:
        min v.T Q v + p.T v
        s.t A v <= b
    Using backtracking line search and Newton's method.
    """

    def centering_objective(t, v):
        return t * qp(v, Q, p) - jnp.sum(jnp.log(b - A @ v))

    def cond(args):
        n_iter, lambda_square, _, _, _ = args
        return (lambda_square / 2 > eps) & (n_iter < max_iter)

    def iter_newton(inps):
        n_iter, _, v, t = inps
        jac, hessian = jax.grad(centering_objective, argnums=1)(t, v), jax.hessian(centering_objective, argnums=1)(t, v)
        descent_step = - jnp.linalg.solve(hessian, jac)
        lambda_square = - jac.T @ descent_step
        _, t_star = backtracking_line_search(v, descent_step, lambda_square)
        v += t_star * descent_step
        return n_iter + 1, lambda_square, v, t

    beta = 0.5
    alpha = 0.1

    def backtracking_line_search(v, descent_step, lambda_square, max_iter=10):
        def cond(inps):
            n_iter, t_p = inps
            return ((centering_objective(t, v + t_p * descent_step) >= centering_objective(t,
                                                                                           v) - alpha * t_p * lambda_square) | (
                        jnp.any(A @ (v + t_p * descent_step) >= b)
                    )) & (n_iter < max_iter)

        def iter_backtracking(inps):
            n_iter, t_p = inps
            t_p *= beta
            return n_iter + 1, t_p

        t_star = jax.lax.while_loop(cond, iter_backtracking, (0, 1.0))
        return t_star

    n_iter, _, v, t = jax.lax.while_loop(cond, iter_newton, (1, 3 * eps, v0, t))
    return n_iter, v


@jax.jit
def barr_method(Q, p, A, b, v0, eps, mu=10, max_iter=100):
    """
    Barrier method for quadratic programs.
    """
    m = b.shape[0]

    def iter_barrier(inps):
        n_iter, v, t = inps
        _, v_star, vs_centering = centering_step(Q, p, A, b, t, v, eps)
        t *= mu
        return n_iter + 1, n_iters_centering, v_star, t

    def cond(args):
        n_iter, _, _, _, t = args
        return (m / t > eps) & (n_iter < max_iter)

    n_iter, n_iters_centering, v_optimal, vs, _ = jax.lax.while_loop(cond, iter_barrier,
                                                                     (1, v0, 1))
    return n_iter, v_optimal
