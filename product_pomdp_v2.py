from itertools import product
from random import choices

import numpy as np
from pomdp_grid import POMDP
from DFA import DFA


class prod_pomdp:

    def __init__(self, pomdp, dfa, secret_dfa_states=[2,4], goal_dfa_states=[0,4]):
        # The width and height of the grid world
        self.width = 6
        self.height = 6
        self.pomdp = pomdp
        self.dfa = dfa
        # Define states
        # # Goals
        self.secret_dfa_states = secret_dfa_states # states in the dfa.
        self.goal_dfa_states = goal_dfa_states # states in the dfa
        self.secret_states = set([])
        self.goal_states = set([])
        self.initial_states= []
        for pomdp_initial_state in self.pomdp.initial_states:
            label = self.pomdp.label_func[pomdp_initial_state]
            label_idx= self.dfa.input_symbols.index(label)
            dfa_initial_state = self.dfa.transition[self.dfa.initial_state][label_idx]
            self.initial_states.append((pomdp_initial_state, dfa_initial_state))
        # Define actions
        self.actions = self.pomdp.actions
        self.selectable_actions = self.pomdp.actions
        self.action_size = len(self.actions)
        self.action_indices = list(range(len(self.actions)))
        # transition probability dictionary
        self.get_transition_incremental()
        self.check_the_transition()
        # Define UAV with sensors
        self.obs_noise = self.pomdp.obs_noise  # the noise of sensors
        self.state_size= len(self.states)
        self.initial_dist = self.get_initial_distribution()
        self.initial_dist_sampling = [1 / len(self.initial_states) for initial_state in self.pomdp.initial_states]
        # Define observations
        self.observations = self.pomdp.observations + [('n', 'n')]
        self.obs_dict = self.get_observation_dictionary()
        self.emiss = self.get_emission_function()
        self.check_emission_function()

    def get_next_supp_with_action(self):
        next_supp = {}
        for st in self.states:
            next_supp[st] = {}
            for act in self.pomdp.actions:
                if st[1] == 0:
                    next_supp[st][act] = ['sink1']  # UAV reaches the goal (nominal agent)
                elif st[1] == 4:
                    next_supp[st][act] = ['sink3']  # adversary is captured (adversary)
                else:
                    next_supp[st][act] = []
                    for pomdp_st_prime in self.pomdp.next_supp[st[0]][act]:
                        input_index = self.dfa.input_symbols.index(self.pomdp.label_func[st[0]])
                        dfa_st_prime = self.dfa.transition[st[1]][input_index]
                        next_supp[st][act].append((pomdp_st_prime, dfa_st_prime))
            # discuss the situation of ending action separately
        return next_supp

    def get_transition_incremental(self):
        states = self.initial_states
        pointer = 0
        trans = {}
        supp = {}
        while pointer < len(states):
            st = states[pointer]
            if st[1] in self.secret_dfa_states:
                self.secret_states.add(st)
            if st[1] in self.goal_dfa_states:
                self.goal_states.add(st)
            trans[st] = {}
            supp[st] = {}
            pointer += 1
            for act in self.pomdp.actions:
                trans[st][act] = {}
                supp[st][act] = set()
                for s_prime in self.pomdp.next_supp[st[0]][act]: # all reachable states in the pomdp.
                    input_index = self.dfa.input_symbols.index(self.pomdp.label_func[st[0]])
                    dfa_st_prime = self.dfa.transition[st[1]][input_index]
                    st_prime = (s_prime, dfa_st_prime)
                    trans[st][act][st_prime] = self.pomdp.transition[st[0]][act][s_prime]
                    if st_prime not in states:
                        states.append(st_prime)
                    supp[st][act].add(st_prime)
        self.states = states
        self.transition = trans
        self.next_supp = supp
        return

    def check_the_transition(self):
        for st in self.states:
            for act in self.actions:
                prob = 0
                for st_prime in self.next_supp[st][act]:
                    prob += self.transition[st][act][st_prime]
                if abs(prob - 1) > 0.01:
                    print("The transition is invalid.", st, act)
        return 0

    def get_observation_dictionary(self):
        obs_dict = {}
        for st in self.states:
            obs_dict[st] = {}
            for act in self.pomdp.actions:
                obs_dict[st][act] = self.pomdp.obs_dict[st[0]][act]
        return obs_dict

    def get_emission_function(self):
        emiss = {}
        for st in self.states:
            emiss[st] = {}
            for act in self.pomdp.actions:
                emiss[st][act] = {}
                for obs in self.observations:
                    if obs in self.obs_dict[st][act]:
                        emiss[st][act][obs] = self.pomdp.emiss[st[0]][act][obs]
                    else:
                        emiss[st][act][obs] = 0
        return emiss

    def check_emission_function(self):
        for st in self.states:
            for act in self.actions:
                prob = 0
                for obs in self.observations:
                    prob += self.emiss[st][act][obs]
                if abs(prob - 1) > 0.01:
                    print("The emission is invalid.", st, act)
        return 0

    def get_initial_distribution(self):
        mu_0 = np.zeros([self.state_size, 1])
        for initial_st in self.initial_states:
            s_0 = self.states.index(initial_st)
            mu_0[s_0, 0] = 1 / len(self.initial_states)
        return mu_0

    def next_state_sampler(self, st, act):
        next_supp = list(self.next_supp[st][act])
        next_prob = [self.transition[st][act][st_prime] for st_prime in next_supp]
        next_state = choices(next_supp, next_prob, k=1)[0]
        return next_state

    def observation_function_sampler(self, st, act):
        observation_set = self.obs_dict[st][act]
        if len(observation_set) == 1:
            return observation_set[0]
        else:
            return choices(observation_set, [1 - self.obs_noise, self.obs_noise], k=1)[0]
