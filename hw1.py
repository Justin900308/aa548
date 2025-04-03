import abc
from typing import Callable
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import functools
import cvxpy as cp
import diffrax
from scipy.integrate import ode

## Problem 1

# class Dynamics(metaclass=abc.ABCMeta):
#     dynamics_func: Callable
#     state_dim: int
#     control_dim: int
#
#     def __init__(self, dynamics_func, state_dim, control_dim):
#         self.dynamics_func = dynamics_func
#         self.state_dim = state_dim
#         self.control_dim = control_dim
#
#     def __call__(self, state, control, time=0):
#         return self.dynamics_func(state, control, time)
#
#
# ss = 2
#
#
# def dynamic_unicycle_ode(state, control, time):
#     x = jnp.array(state)
#     u = jnp.array(control)
#     x_dot = jnp.array([x[3] * jnp.cos(x[2]),
#                        x[3] * jnp.sin(x[2]),
#                        u[0],
#                        u[1]])
#
#     ss = 2
#     return x_dot
#
#
# state_dim = 4
# control_dim = 2
# continuous_dynamics = Dynamics(dynamic_unicycle_ode, state_dim, control_dim)
# ss = 2
#
#
# def euler_integrate(dynamics, dt):
#     def integrator(x, u, t):
#         x_dot = dynamics(x, u, t)
#         xtp1 = x + x_dot * dt
#         return xtp1
#
#     return integrator
#
#
# ## I choose to use RK4 (https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
# def runge_kutta_integrator(dynamics, dt=0.1):
#     def integrator(x, u, t):
#         k1 = dynamics(x, u, t)
#         k2 = dynamics(x + dt * k1 / 2, u, t + dt / 2)
#         k3 = dynamics(x + dt * k2 / 2, u, t + dt / 2)
#         k4 = dynamics(x + dt * k3, u, t + dt)
#         xtp1 = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
#         return xtp1
#
#     return integrator
#
#
# ss = 2
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
# ss = 2
#
#
# def simulate(dynamics, initial_state, controls, dt):
#     t = 0
#     x_t = initial_state
#     x_all = initial_state
#     for u in controls:
#         x_tp1 = dynamics(x_t, u, t)
#         x_all = jnp.vstack((x_all, x_tp1))
#         x_t = x_tp1
#         t = t + dt
#
#     return x_all


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


# ss = 2
#
# state_dim = continuous_dynamics.state_dim
# control_dim = continuous_dynamics.control_dim
# N = 1000  # Number of trajectories
# n_time_steps = 50
# dt = 0.1
# initial_states = jnp.array(np.random.randn(N, state_dim))
# controls = jnp.array(np.random.randn(N, n_time_steps, control_dim))
#
# trajs_euler = jax.vmap(simulate, in_axes=[None, 0, 0, None])(discrete_dynamics_euler, initial_states, controls, dt)
# trajs_rk4 = jax.vmap(simulate, in_axes=[None, 0, 0, None])(discrete_dynamics_rk, initial_states, controls, dt)
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
