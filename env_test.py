import numpy as np
import pickle
from grid_world_example import Environment
from pomdp_grid import POMDP
from product_pomdp import prod_pomdp
from random import choices
from random import choice

grid_world = Environment()
pomdp = POMDP()
prod_pomdp = prod_pomdp()


def sample_random_data(M, T, type='mix'):
    st_data = []
    act_data = []
    y_data = []
    for m in range(M):
        st_list = []
        act_list = []
        y = []
        # start from initial state
        if type == 'mix':
            st = choices(prod_pomdp.initial_states, prod_pomdp.initial_dist_sampling, k=1)[0]
        elif type == '1':
            st = prod_pomdp.initial_states[1]
        elif type == '0':
            st = prod_pomdp.initial_states[0]
        else:
            raise ValueError('Invalid type value.')
        # Sample sensing action
        act = choice(prod_pomdp.selectable_actions)
        act_list.append(act)
        # Get the observation of initial state
        obs = prod_pomdp.observation_function_sampler(st, act)
        y.append(obs)

        for t in range(T - 1):
            st_list.append(st)
            # sample the next state
            st = prod_pomdp.next_state_sampler(st, act)
            # Sample sensing action
            act = choice(prod_pomdp.selectable_actions)
            act_list.append(act)
            # Add the observation
            obs = prod_pomdp.observation_function_sampler(st, act)
            y.append(obs)
        # Add the ending action
        act = 'e'
        act_list.append(act)
        st = prod_pomdp.next_state_sampler(st, act)
        st_list.append(st)
        obs = prod_pomdp.observation_function_sampler(st, act)
        y.append(obs)

        st_data.append(st_list)
        y_data.append(y)
        act_data.append(act_list)
    return st_data, y_data, act_data


def display_states_from_data(data):
    M = len(data)
    for k in range(M):
        print(data[k])
    return 0


M = 100  # number of sampled trajectories
T = 20  # length of a trajectory
# print(prod_pomdp.next_state_sampler('sink2', 'e'))
s_data, y_data, a_data = sample_random_data(M, T)
display_states_from_data(s_data)
print('#' * 100)
display_states_from_data(y_data)
print('#' * 100)
display_states_from_data(a_data)
