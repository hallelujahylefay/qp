import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
max_iter = 100


@jax.jit
def qp(v, Q, p):
    """
    Quadratic program objective function
    """
    return v.T @ Q @ v + p.T @ v


@jax.jit
def centering_step(Q, p, A, b, t, v0, eps, max_iter=max_iter):
    """
    Centering step of the barrier method for quadratic programs:
        min v.T Q v + p.T v
        s.t A v <= b
    Using backtracking line search and Newton's method.
    This code is not optimized for memory usage, there is no need to store all the iterates.
    One can remove vs variable.
    Moreover, since one knows the gradient and hessian for the QP. barrier objective, no need to use jax.grad and jax.hessian.
    """

    def centering_objective(t, v):
        return t * qp(v, Q, p) - jnp.sum(jnp.log(b - A @ v))

    def cond(args):
        n_iter, lambda_square, _, _, _ = args
        return (lambda_square / 2 > eps) & (n_iter < max_iter)

    def iter_newton(inps):
        n_iter, _, v, vs, t = inps
        vs = vs.at[n_iter].set(v)
        jac, hessian = jax.grad(centering_objective, argnums=1)(t, v), jax.hessian(centering_objective, argnums=1)(t, v)
        descent_step = - jnp.linalg.solve(hessian, jac)
        lambda_square = - jac.T @ descent_step
        _, t_star = backtracking_line_search(v, descent_step, lambda_square)
        v += t_star * descent_step
        return n_iter + 1, lambda_square, v, vs, t

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

    vs0 = jnp.zeros((max_iter, *v0.shape))
    n_iter, _, v, vs, t = jax.lax.while_loop(cond, iter_newton, (1, 3 * eps, v0, vs0, t))
    return n_iter, v, vs


@jax.jit
def barr_method(Q, p, A, b, v0, eps, mu=10, max_iter=max_iter):
    """
    Barrier method for quadratic programs.
    This code is not optimized for memory usage, there is no need to store all the iterates.
    One can remove vs variable.
    """
    m = b.shape[0]

    def iter_barrier(inps):
        n_iter, n_iters_centering, v, vs, t = inps
        n_iter_centering, v_star, vs_centering = centering_step(Q, p, A, b, t, v, eps)
        n_iters_centering = n_iters_centering.at[n_iter].set(n_iter_centering)
        vs = vs.at[n_iter].set(vs_centering)
        t *= mu
        return n_iter + 1, n_iters_centering, v_star, vs, t

    def cond(args):
        n_iter, _, _, _, t = args
        return (m / t > eps) & (n_iter < max_iter)

    vs0 = jnp.zeros((max_iter, max_iter, *v0.shape))
    vs0 = vs0.at[0].set(jnp.full((max_iter, *v0.shape), v0))
    n_iter, n_iters_centering, v_optimal, vs, _ = jax.lax.while_loop(cond, iter_barrier,
                                                                     (1, jnp.zeros((max_iter,), dtype=int), v0, vs0, 1))
    return n_iter, n_iters_centering, v_optimal, vs
