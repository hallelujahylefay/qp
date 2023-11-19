import jax
import jax.numpy as jnp
from qp.lasso import solve_lasso
from qp.timeit import timeit
import matplotlib.pyplot as plt
import numpy as np

N = 5  # number of lasso problems
OP_key = jax.random.PRNGKey(0)  # seed
keys = jax.random.split(OP_key, N)

shape = (15, 40)
mus = jnp.array([2, 10, 15, 50, 75, 100, 150, 200])
penalization = 10
distrib = jax.random.uniform

wnorm = np.zeros((len(mus), N))  # for storage purpose
las_obj = np.zeros((len(mus), N))

for idx, mu in enumerate(mus):
    @timeit
    @jax.vmap
    def wrapper_lasso_barr_method(key):
        X, y = distrib(key, shape=shape), distrib(key, shape=(shape[0],))
        return solve_lasso(X, y, penalization=penalization, mu=mu, eps=1e-6)


    n_iters, n_iters_centering, _, primals, values_for_dual_objective, lasso_objective_values = wrapper_lasso_barr_method(
        keys)

    for i in range(N):
        iteration = n_iters[i]
        reconstructed_values_for_dual_objective = jnp.concatenate(
            [(values_for_dual_objective[i][:iteration][j][:n_iters_centering[i][j]]) for j in
             range(iteration)])

        min = jnp.min(reconstructed_values_for_dual_objective)
        plt.semilogy(range(len(reconstructed_values_for_dual_objective)), reconstructed_values_for_dual_objective - min)

    plt.title(f"{N} lasso problems w/ shape {shape[0], shape[1]}, semilog error, mu={mu}")
    plt.savefig(f"./plots/lasso_qp_{mu}", dpi=500)
    plt.close()

    for i in range(N):
        iteration = n_iters[i]
        reconstructed_values_for_primal = jnp.linalg.norm(jnp.concatenate(
            [(primals[i][:iteration][j][:n_iters_centering[i][j]]) for j in
             range(1, n_iters[i])]),
            axis=-1)[-1]
        wnorm[idx][i] = reconstructed_values_for_primal[-1]
        plt.semilogy(range(len(reconstructed_values_for_primal)), reconstructed_values_for_primal)

    plt.title(f"{N} lasso problems w/ shape {shape[0], shape[1]}, norm w, mu={mu}")
    plt.savefig(f"./plots/lasso_qp_norm_{mu}", dpi=500)
    plt.close()

    for i in range(N):
        iteration = n_iters[i]
        reconstructed_values_for_lasso_objective_values = jnp.concatenate(
            [(lasso_objective_values[i][:iteration][j][:n_iters_centering[i][j]]) for j in
             range(iteration)])
        las_obj[idx][i] = reconstructed_values_for_lasso_objective_values[-1]
        plt.semilogy(range(len(reconstructed_values_for_lasso_objective_values)),
                     reconstructed_values_for_lasso_objective_values)

    plt.title(f"{N} lasso problems w/ shape {shape[0], shape[1]}, lasso objective, mu={mu}")
    plt.savefig(f"./plots/lasso_qp_lasso_objective_{mu}", dpi=500)
    plt.close()

for i in range(N):
    plt.plot(mus, wnorm[:, i])
plt.title(f"{N} lass problems w/ shape {shape[0], shape[1]}, w norm w.r.t mu")
plt.savefig(f"./plots/lass_qp_norm_mus", dpi=500)
plt.close()

for i in range(N):
    plt.plot(mus, las_obj[:, i])
plt.title(f"{N} lass problems w/ shape {shape[0], shape[1]}, lasso obj. w.r.t mu")
plt.savefig(f"./plots/lass_qp_obj_mus", dpi=500)
plt.close()
