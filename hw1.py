import abc
from typing import Callable
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import functools
import cvxpy as cp
from scipy import signal


############################## P1

class Dynamics(metaclass=abc.ABCMeta):
    dynamics_func: Callable
    state_dim: int
    control_dim: int

    def __init__(self, dynamics_func, state_dim, control_dim):
        self.dynamics_func = dynamics_func
        self.state_dim = state_dim
        self.control_dim = control_dim

    def __call__(self, state, control, time=0):
        return self.dynamics_func(state, control, time)


ss = 2


def dynamic_unicycle_ode(state, control, time):
    x = jnp.array(state)
    u = jnp.array(control)
    x_dot = jnp.array([x[3] * jnp.cos(x[2]),
                       x[3] * jnp.sin(x[2]),
                       u[0],
                       u[1]])

    ss = 2
    return x_dot


state_dim = 4
control_dim = 2
continuous_dynamics = Dynamics(dynamic_unicycle_ode, state_dim, control_dim)
ss = 2


def euler_integrate(dynamics, dt):
    def integrator(x, u, t):
        x_dot = dynamics(x, u, t)
        xtp1 = x + x_dot * dt
        return xtp1

    return integrator


## I choose to use RK4 (https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
def runge_kutta_integrator(dynamics, dt):
    def integrator(x, u, t):
        k1 = dynamics(x, u, t)
        k2 = dynamics(x + dt * k1 / 2, u, t + dt / 2)
        k3 = dynamics(x + dt * k2 / 2, u, t + dt / 2)
        k4 = dynamics(x + dt * k3, u, t + dt)
        xtp1 = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xtp1

    return integrator


# Example usage of the integrators

dt = 0.1  # Default timestep size

discrete_dynamics_euler = Dynamics(
    euler_integrate(continuous_dynamics, dt), state_dim, control_dim
)
discrete_dynamics_rk = Dynamics(
    runge_kutta_integrator(continuous_dynamics, dt), state_dim, control_dim
)


def simulate(dynamics, initial_state, controls, dt):
    t = 0
    x_t = initial_state
    x_all = initial_state
    for u in controls:
        x_tp1 = dynamics(x_t, u, t)  # discrete dynamics
        x_all = jnp.vstack((x_all, x_tp1))
        x_t = x_tp1
        t = t + dt

    return x_all


# initial_state = jnp.array([0.0, 0.0, 0.0, 0.0])
# control = jnp.array([2.0, 1.0])  # Constant control over the duration.
# duration = 5.0
# dts = [0.01, 0.5]
#
# for dt in dts:
#     num_steps = int(duration / dt)
#     controls = [control] * num_steps
#
#     # Construct the discrete dynamics for the given timestep
#     discrete_dynamics_euler = Dynamics(
#         euler_integrate(continuous_dynamics, dt), state_dim, control_dim
#     )
#     discrete_dynamics_rk = Dynamics(
#         runge_kutta_integrator(continuous_dynamics, dt), state_dim, control_dim
#     )
#
#     # Simulate dynamics
#     xs_euler = simulate(discrete_dynamics_euler, initial_state, controls, dt)
#     xs_rk = simulate(discrete_dynamics_rk, initial_state, controls, dt)
#     xs_euler = np.array(xs_euler)
#
#     plt.plot(xs_euler[:, 0], xs_euler[:, 1], label=f"dt = {dt} Euler")
#     plt.plot(xs_rk[:, 0], xs_rk[:, 1], label=f"dt = {dt} RK")
#     plt.legend()
#
# plt.grid(alpha=0.4)
# plt.axis("equal")
# plt.xlabel("x [m]")
# plt.ylabel("y [m]")
# plt.show()
#
#
# ss = 2
#
# state_dim = continuous_dynamics.state_dim
# control_dim = continuous_dynamics.control_dim
# N = 1000  # Number of trajectories
# n_time_steps = 50
# dt = 0.1
# initial_states = jnp.array(np.random.randn(N, state_dim))
# controls = jnp.array(np.random.randn(N, n_time_steps, control_dim))
# sim_jit = jax.jit(simulate, static_argnums=0)
# trajs_euler = jax.vmap(sim_jit, in_axes=[None, 0, 0, None])(discrete_dynamics_euler, initial_states, controls, dt)
# trajs_rk4 = jax.vmap(sim_jit, in_axes=[None, 0, 0, None])(discrete_dynamics_rk, initial_states, controls, dt)
# trajs_euler = np.array(trajs_euler)
# trajs_rk4 = np.array(trajs_rk4)
#
# # plot to visualize all the trajectories
# for i in range(N):
#     traj_euler_i = trajs_euler[i, :, :]
#     traj_rk4_i = trajs_rk4[i, :, :]
#     plt.plot(traj_euler_i[:, 0], traj_euler_i[:, 1], label=f"dt = {dt} Euler")
#     plt.plot(traj_rk4_i[:, 0], traj_rk4_i[:, 1], label=f"dt = {dt} RK4")
# # plt.legend()
# plt.show()


############################## P2

# def linearize_unicycle_continuous_time_analytic(state, control, time):
#     x0 = jnp.array(state)
#     u0 = jnp.array(control)
#
#     A = jnp.array([[0, 0, -x0[3] * jnp.sin(x0[2]), jnp.cos(x0[2])],
#                    [0, 0, x0[3] * jnp.cos(x0[2]), jnp.sin(x0[2])],
#                    [0, 0, 0, 0],
#                    [0, 0, 0, 0]])
#     B = jnp.array([[0, 0],
#                    [0, 0],
#                    [1, 0],
#                    [0, 1]])
#     C = jnp.array([x0[3] * jnp.cos(x0[2]), x0[3] * jnp.sin(x0[2]), u0[0], u0[1]])
#     C = C - A @ x0 - B @ u0
#     return A, B, C  # update this line
#
#
# def linearize_autodiff(continuous_dynamics, state, control, time):
#     x0 = jnp.array(state)
#     u0 = jnp.array(control)
#     t0 = jnp.array(time)
#     A = jax.jacobian(lambda x: continuous_dynamics(x, u0, t0))(x0)
#     B = jax.jacobian(lambda u: continuous_dynamics(x0, u, t0))(u0)
#     C = jnp.array([x0[3] * jnp.cos(x0[2]), x0[3] * jnp.sin(x0[2]), u0[0], u0[1]])
#     C = C - A @ x0 - B @ u0
#     return A, B, C  # update this line
#
#
# # test code:
# state = jnp.array([0.0, 0.0, jnp.pi / 4, 2.])
# control = jnp.array([0.1, 1.])
# time = 0.
# dt = 0.1
# A_autodiff, B_autodiff, C_autodiff = linearize_autodiff(continuous_dynamics, state, control, time)
# A_analytic, B_analytic, C_analytic = linearize_unicycle_continuous_time_analytic(state, control, time)
# print('A matrices match:', jnp.allclose(A_autodiff, A_analytic))
# print('B matrices match:', jnp.allclose(B_autodiff, B_analytic))
# print('C matrices match:', jnp.allclose(C_autodiff, C_analytic))
#
# # Example usage of the integrators
#
# dt = 0.1  # Default timestep size
#
# discrete_dynamics_euler = Dynamics(
#     euler_integrate(continuous_dynamics, dt), state_dim, control_dim
# )
# discrete_dynamics_rk = Dynamics(
#     runge_kutta_integrator(continuous_dynamics, dt), state_dim, control_dim
# )
#
#
# def discete_autodiff(discrete_dynamic, state, control, time):
#     x_t = jnp.array(state)
#     u_t = jnp.array(control)
#     t = jnp.array(time)
#     A_dis = jax.jacobian(lambda x: discrete_dynamic(x, u_t))(x_t)
#     B_dis = jax.jacobian(lambda u: discrete_dynamic(x_t, u))(u_t)
#
#     return A_dis, B_dis
#
#
# ## sanity check with scipy
# C = np.eye(4)
# D = np.zeros((4, 2))
# A = np.array(A_analytic)
# B = np.array(B_analytic)
# sys = signal.StateSpace(A, B, C, D)
# sysd = sys.to_discrete(dt)
# Ad = sysd.A
# Bd = sysd.B
#
# A_autodiff_euler, B_autodiff_euler = discete_autodiff(discrete_dynamics_euler, state, control, time)
# A_autodiff_rk4, B_autodiff_rk4 = discete_autodiff(discrete_dynamics_rk, state, control, time)
#
# key = jax.random.PRNGKey(42)  # Set a fixed seed
# n_samples = 1000
# state_dim = 4  # 4-dimensional state
# ctrl_dim = 2  # 2-dimensional control
# time = 0.0
# random_states = jax.random.normal(key, shape=(n_samples, state_dim))
# random_controls = jax.random.normal(key, shape=(n_samples, ctrl_dim))
#
# linear_euler = jax.vmap(discete_autodiff, in_axes=[None, 0, 0, None])(discrete_dynamics_euler, random_states,
#                                                                       random_controls, time)
# linear_rk4 = jax.vmap(discete_autodiff, in_axes=[None, 0, 0, None])(discrete_dynamics_rk, random_states,
#                                                                     random_controls, time)
# linear_euler_A = np.array(linear_euler[0])
# linear_euler_B = np.array(linear_euler[1])
# linear_rk4_A = np.array(linear_rk4[0])
# linear_rk4_B = np.array(linear_rk4[1])

############################## P3
