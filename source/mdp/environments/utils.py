import numpy as np

class bound():
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __contains__(self, key):
        return key >= self.lower_bound and key <= self.upper_bound


def func_x_dot(v):
    return v

def func_v_dot(F, m_c, m_p, l, theta_dot, theta, omega_dot):
    num = F + m_p * l * ((theta_dot**2)*np.sin(theta) - omega_dot*np.cos(theta))
    den = m_c + m_p
    return num/den

def func_theta_dot(omega):
    return omega

def func_omega_dot(g, theta, F, m_p, m_c, omega, l):
    num = g * np.sin(theta) + np.cos(theta) * (-F-m_p*l*(omega**2)*np.sin(theta)) / (m_c+m_p)
    den  = l*(4.0/3 - m_p * ((np.cos(theta))**2) / (m_c + m_p))
    return num/den

def func_state_tp1(state_t, deltat, state_t_dot):
    return state_t + deltat * state_t_dot