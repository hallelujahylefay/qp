import jax
import jax.numpy as jnp


@jax.jit
def centering_step(Q, p, A, b, t, v0, eps, max_iter=1000):
    """
    Centering step of the barrier method for quadratic programs:
        min v.T Q v + p.T v
        s.t A v <= b
    Using backtracking line search and Newton's method.
    This code is not optimized for memory usage, there is no need to store all the iterates.
    """

    def objective(v):
        return v.T @ Q @ v + p.T @ v

    def centering_objective(t, v):
        return t * objective(v) - jnp.sum(jnp.log(b - A @ v))

    def cond(args):
        n_iter, lambda_square, _, _, _ = args
        return (lambda_square / 2 > eps) & (n_iter <= max_iter)

    def iter_newton(inps):
        n_iter, _, v, vs, t = inps
        _centering_objective = lambda _v: centering_objective(t, _v)

        jac, hessian = jax.jacfwd(_centering_objective)(v), jax.hessian(_centering_objective)(v)
        descent_step = - jnp.linalg.inv(hessian) @ jac
        lambda_square = - jac.T @ descent_step
        t_star = backtracking_line_search(v, descent_step, lambda_square)
        v += t_star * descent_step
        vs = vs.at[n_iter].set(v)
        return n_iter + 1, lambda_square, v, vs, t

    beta = 0.9
    alpha = 0.1

    def backtracking_line_search(v, descent_step, lambda_square):
        def cond(t_p):
            return centering_objective(t, v + t_p * descent_step) >= centering_objective(t,
                                                                                         v) + alpha * t_p * lambda_square

        def iter_backtracking(t_p):
            return t_p * beta

        t_star = jax.lax.while_loop(cond, iter_backtracking, 1.0)
        return t_star

    vs0 = jnp.zeros((max_iter, *v0.shape))
    vs0 = vs0.at[0].set(v0)
    _, _, v, vs, t = jax.lax.while_loop(cond, iter_newton, (1, 3 * eps, v0, vs0, t))
    return v, vs


@jax.jit
def barr_method(Q, p, A, b, v0, eps, mu=10, max_iter=1000):
    """
    Barrier method for quadratic programs.
    This code is not optimized for memory usage, there is no need to store all the iterates.
    """
    m = b.shape[0]

    def iter_barrier(inps):
        n_iter, v, vs, t = inps
        v_star, _ = centering_step(Q, p, A, b, t, v, eps)
        vs = vs.at[n_iter].set(v_star)
        t *= mu
        return n_iter + 1, v_star, vs, t

    def cond(args):
        n_iter, _, _, t = args
        return (m / t > eps) & (n_iter <= max_iter)

    vs0 = jnp.zeros((max_iter, *v0.shape))
    vs0 = vs0.at[0].set(v0)
    _, v_optimal, vs, _ = jax.lax.while_loop(cond, iter_barrier, (1, v0, vs0, 1))
    return v_optimal, vs
