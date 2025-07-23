from itertools import product
from grid_world_example import Environment

grid_world = Environment()


class POMDP:

    def __init__(self):
        # The width and height of the grid world
        self.width = grid_world.width
        self.height = grid_world.height
        # Define states
        # Here tau = 0 (L(s) = !t) means nominal agent and tau = 1 (L(s) = t) means adversary
        self.states = [(gr_st, uav_st, tau) for gr_st, uav_st, tau in
                       product(grid_world.gr_states, grid_world.uav_states, [0, 1])]
        # self.state_indices = list(range(len(self.states)))
        # self.state_size = len(self.states)
        # Define initial state
        self.initial_states = [(grid_world.gr_initial_state, grid_world.uav_initial_state, 0),
                               (grid_world.gr_initial_state, grid_world.uav_initial_state, 1)]
        # self.initial_state_idx = self.states.index(self.initial_state)
        # Define actions
        self.actions = grid_world.uav_actions
        self.action_size = len(self.actions)
        self.action_indices = list(range(len(self.actions)))
        # transition probability dictionary
        self.next_supp = self.get_next_supp_with_action()
        self.transition = self.get_transition()
        self.check_the_transition()
        # Define UAV with sensors
        self.obs_noise = 0.1  # the noise of sensors
        # Define observations
        self.observations = [(gr_st, uav_st) for gr_st, uav_st in
                       product((grid_world.gr_states + ['n']), grid_world.uav_states)]
        self.obs_dict = self.get_observation_dictionary()
        self.emiss = self.get_emission_function()
        self.check_emission_function()
        # Define the atomic propositions
        self.atom_prop = ['t', 'a', 'p']  # a for goal, p for capture, t indicates adversary
        # Define the labeling function
        self.label_func = self.get_label_function()

    def get_next_supp_with_action(self):
        next_supp = {}
        for st in self.states:
            next_supp[st] = {}
            for act in self.actions:
                if st[2] == 0:
                    gr_next_supp = list(grid_world.transition_n[st[0]].keys())
                elif st[2] == 1:
                    gr_next_supp = list(grid_world.transition_a[st[0]].keys())
                else:
                    raise ValueError('Invalid type value.')
                uav_next_supp = list(grid_world.uav_transition[st[1]][act].keys())
                next_supp[st][act] = [(gr_st, uav_st, st[2]) for gr_st, uav_st in product(gr_next_supp, uav_next_supp)]
        return next_supp

    def get_transition(self):
        trans = {}
        for st in self.states:
            trans[st] = {}
            for act in self.actions:
                trans[st][act] = {}
                for st_prime in self.next_supp[st][act]:
                    if st[2] == 0:
                        trans[st][act][st_prime] = grid_world.transition_n[st[0]][st_prime[0]] * \
                                                   grid_world.uav_transition[st[1]][act][st_prime[1]]
                    elif st[2] == 1:
                        trans[st][act][st_prime] = grid_world.transition_a[st[0]][st_prime[0]] * \
                                                   grid_world.uav_transition[st[1]][act][st_prime[1]]
                    else:
                        raise ValueError('Invalid type value.')
        return trans

    def check_the_transition(self):
        for st in self.states:
            for act in self.actions:
                prob = 0
                for st_prime in self.next_supp[st][act]:
                    prob += self.transition[st][act][st_prime]
                if abs(prob - 1) > 0.01:
                    print("The transition is invalid.", self.transition[st][act])
        return 0

    def get_observation_dictionary(self):
        obs_dict = {}
        for st in self.states:
            obs_dict[st] = {}
            for act in self.actions:
                if grid_world.neighbor[st[0]][st[1]]:
                    obs_dict[st][act] = [(st[0], st[1]), ('n', st[1])] #  observation with detection and non detection.
                else:
                    obs_dict[st][act] = [('n', st[1])]
        return obs_dict

    def get_emission_function(self):
        emiss = {}
        for st in self.states:
            emiss[st] = {}
            for act in self.actions:
                emiss[st][act] = {obs: 0 for obs in self.observations}  # Initialize all observations to 0
                obs_possible =  self.obs_dict[st][act]
                if grid_world.neighbor[st[0]][st[1]]:

                    emiss[st][act][obs_possible[0]] = 1 - self.obs_noise # JF: modified observation
                    emiss[st][act][obs_possible[1]] = self.obs_noise
                else:
                    emiss[st][act][obs_possible[0]] = 1
        return emiss
    # def get_emission_function(self):
    #     emiss = {}
    #     for st in self.states:
    #         emiss[st] = {}
    #         for act in self.actions:
    #             emiss[st][act] = {}
    #             for obs in self.observations:
    #                 if obs in self.obs_dict[st][act]:
    #                     if grid_world.neighbor[st[0]][st[1]]:
    #                         if st[0] == obs[0]:
    #                             emiss[st][act][obs] = 1 - self.obs_noise # JF: modified observation
    #                         else:
    #                             emiss[st][act][obs] = self.obs_noise
    #                     else:
    #                         emiss[st][act][obs] = 1
    #                 else:
    #                     emiss[st][act][obs] = 0
    #     return emiss

    def check_emission_function(self):
        for st in self.states:
            for act in self.actions:
                prob = 0
                for obs in self.observations:
                    prob += self.emiss[st][act][obs]
                if abs(prob - 1) > 0.01: # check if the sum of probabilities is one.
                    print("The emission is invalid.", self.emiss[st][act])
        return 0

    def get_label_function(self):
        lab = {}
        for st in self.states:
            lab[st] = []
            if st[2] == 1: # adversary
                lab[st] += self.atom_prop[0]
            if st[1] in grid_world.uav_goal: # reaching goal
                lab[st] += self.atom_prop[1]
            if st[0] == st[1]: # capture
                lab[st] += self.atom_prop[2]
        return lab
