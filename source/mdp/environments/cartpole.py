import numpy as np
from typing import Tuple
from .skeleton import Environment
from .utils import bound
from .utils import func_omega_dot, func_state_tp1, func_theta_dot, func_v_dot, func_x_dot


class Cartpole(Environment):
    """
    The cart-pole environment as described in the 687 course material. This
    domain is modeled as a pole balancing on a cart. The agent must learn to
    move the cart forwards and backwards to keep the pole from falling.

    Actions: left (0) and right (1)
    Reward: 1 always

    Environment Dynamics: See the work of Florian 2007
    (Correct equations for the dynamics of the cart-pole system) for the
    observation of the correct dynamics.
    """

    def __init__(self):
        # TODO: properly define the variables below
        self._name = "Cartpole"
        self._action = None
        self._reward = 0
        self._isEnd = False
        self._gamma = 1.0

        # define the state # NOTE: you must use these variable names
        self._x = 0.0  # horizontal position of cart
        self._v = 0.0  # horizontal velocity of the cart
        self._theta = 0.0  # angle of the pole
        self._dtheta = 0.0  # angular velocity of the pole

        # dynamics
        self._g = 9.8  # gravitational acceleration (m/s^2)
        self._mp = 0.1  # pole mass kg
        self._mc = 1.0  # cart mass kg
        self._l = 0.5  # (1/2) * pole length m
        self._dt = 0.02  # timestep
        # self._ttt = 20.0  # time to terminate
        
        # # Environment Properties
        # self._max_force = 10.0
        # self._fail_angle = np.pi / 12
        # # self._bounds = bound(-3, 3)
        # self._bound = 3.0
        
        self._t = 0.0  # total time elapsed  NOTE: USE must use this variable

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def action(self) -> int:
        return self._action

    @property
    def isEnd(self) -> bool:
        return self.terminal()
        
    @property
    def state(self) -> np.ndarray:
        # TODO
        return np.array([self._x ,self._v ,self._theta ,self._dtheta])

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the next state of the pendulum using the euler approximation to the dynamics
        """
        # TODO
        if action == 0:
            F_t = -10.0
        else:
            F_t = 10.0
        x_t, v_t, theta_t, omega_t = state
        delta_t = self._dt

        x_t_dot = func_x_dot(v_t)
        theta_t_dot = omega_t
        omega_t_dot = func_omega_dot(self._g, theta_t, F_t, self._mp, self._mc, omega_t, self._l)
        v_t_dot = func_v_dot(F_t, self._mc, self._mp, self._l, theta_t_dot, theta_t, omega_t_dot)

        state_t_dot = np.array([x_t_dot, v_t_dot, theta_t_dot, omega_t_dot])

        # Update self to the new state

        state_tp1 = state + delta_t * state_t_dot

        return state_tp1


    def R(self, state: np.ndarray, action: int, nextState: np.ndarray) -> float:
        return 1.0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        takes one step in the environment and returns the next state, reward, and if it is in the terminal state
        """
        state_tp1 = self.nextState(self.state, action)
        R_t = self.R(self.state, action, state_tp1)

        # Update the variables of the class
        self._action = action
        self._x, self._v, self._theta, self._dtheta = state_tp1
        self._reward = R_t
        self._t += self._dt
        self._isEnd = self.terminal()

        return (state_tp1, R_t, self._isEnd)

    def reset(self) -> None:
        """
        resets the state of the environment to the initial configuration
        """
        self._action = None
        self._reward = 0
        self._isEnd = False
        self._gamma = 1.0
        self._t = 0.0

        # define the state # NOTE: you must use these variable names
        self._x = 0.0  # horizontal position of cart
        self._v = 0.0  # horizontal velocity of the cart
        self._theta = 0.0  # angle of the pole
        self._dtheta = 0.0  # angular velocity of the pole

    def terminal(self) -> bool:
        """
        The episode is at an end if:
            time is greater that 20 seconds
            pole falls |theta| > (pi/12.0)
            cart hits the sides |x| >= 3
        """
        t1 = (abs(self._theta) > np.pi/12.0)
        t2 = (abs(self._x) >= 3)
        t3 = (self._t > 20)
        return t1 or t2 or t3

