from random import choices
import numpy as np


class Environment:

    def __init__(self):
        # The width and height of the grid world
        self.width = 6
        self.height = 6
        # parameter which controls environment noise
        self.stoPar = 0.1
        # parameter which controls observation noise
        self.obs_noise = 0.1
        # Define obstacles
        self.obstacles = [(2, 1), (5, 1), (0, 2), (3, 3)]
        # Define states
        self.whole_states = [(i, j) for i in range(self.width) for j in range(self.height)]
        self.gr_states = list(set(self.whole_states) - set(self.obstacles))
        self.uav_states = self.gr_states
        self.gr_state_indices = list(range(len(self.gr_states)))
        # self.state_size = len(self.gr_states)
        # Define initial state
        self.gr_initial_state = (0, 3)
        self.uav_initial_state = (3, 0)
        # self.initial_state_dis = self.get_initial_distribution()
        # Define actions
        self.gr_actions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
        self.uav_actions = self.gr_actions
        # self.action_size = len(self.actions)
        # self.action_indices = list(range(len(self.actions)))
        # Goals
        self.goals_n = [(5, 5)]  # The goal of nominal agent
        self.goals_a = [(5, 0)]  # The goal of adversary agent
        self.uav_goal = [(0, 5)]
        # transition probability dictionary
        self.gr_transition = self.get_transition('g')
        self.uav_transition = self.get_transition('u')
        # reward dictionary
        self.goal_reward = 1
        self.rewards_n = self.get_rewards(self.goals_n)  # The rewards of nominal agent
        self.rewards_a = self.get_rewards(self.goals_a)  # The rewards of adversary
        # discounting factor
        self.gamma = 0.9
        self.opt_policy_n = self.value_iteration(self.rewards_n)[1]  # The optimal policy of nominal agent
        self.opt_policy_a = self.value_iteration(self.rewards_a)[1]  # The optimal policy of adversary
        # calculate the transition of ground robots under the optimal policy
        self.transition_n = self.get_states_transition('n')
        self.transition_a = self.get_states_transition('a')
        # neighbor function
        self.neighbor = self.get_neighbor_function()

    def get_neighbor_function(self):
        neigh = {}
        for gr_st in self.gr_states:
            neigh[gr_st] = {}
            for uav_st in self.uav_states:
                dist = np.sqrt((gr_st[0] - uav_st[0])**2 + (gr_st[1] - uav_st[1])**2)
                if 2 > dist >= 0:
                    neigh[gr_st][uav_st] = True
                else:
                    neigh[gr_st][uav_st] = False
        return neigh

    def get_rewards(self, goals):
        rewards = {}
        for st in self.gr_states:
            rewards[st] = {}
            for act in self.gr_actions:
                if st in goals:
                    rewards[st][act] = self.goal_reward
                else:
                    rewards[st][act] = 0
        return rewards


    def complementary_actions(self, act):
        # Use to find out stochastic transitions, if it stays, no stochasticity, if other actions, return possible stochasticity directions.
        if act == (0, 0):
            return []
        elif act[0] == 0:
            return [(1, 0), (-1, 0)]
        else:
            return [(0, 1), (0, -1)]

    def check_inside(self, st, agent):
        if agent == 'g':
            states = self.gr_states
        elif agent == 'u':
            states = self.uav_states
        else:
            raise ValueError('Invalid agent parameter.')
        # If the state is valid or not
        if st in states:
            return True
        return False

    def get_transition(self, agent):
        if agent == 'g':
            states = self.gr_states
            actions = self.gr_actions
        elif agent == 'u':
            states = self.uav_states
            actions = self.uav_actions
        else:
            raise ValueError('Invalid agent parameter.')
        # Constructing transition function trans[state][action][next_state] = probability
        stoPar = self.stoPar
        trans = {}
        for st in states:
            trans[st] = {}
            for act in actions:
                if act == (0, 0):
                    trans[st][act] = {}
                    trans[st][act][st] = 1
                else:
                    trans[st][act] = {}
                    trans[st][act][st] = 0
                    tempst = tuple(np.array(st) + np.array(act))
                    if self.check_inside(tempst, agent):
                        trans[st][act][tempst] = 1 - 2 * stoPar
                    else:
                        trans[st][act][st] += 1 - 2 * stoPar
                    for act_ in self.complementary_actions(act):
                        tempst_ = tuple(np.array(st) + np.array(act_))
                        if self.check_inside(tempst_, agent):
                            trans[st][act][tempst_] = stoPar
                        else:
                            trans[st][act][st] += stoPar
        # self.check_trans(trans)
        return trans

    def check_trans(self, trans):
        # Check if the transitions are constructed correctly
        for st in trans.keys():
            for act in trans[st].keys():
                if abs(sum(trans[st][act].values()) - 1) > 0.01:
                    print("st is:", st, "act is:", act, "sum is:", sum(trans[st][act].values()))
                    return False
        print("Transition is correct")
        return True

    def extract_policy(self, rewards, V):
        """
        Extracts the optimal policy given the optimal value function.

        :param rewards: Dictionary with keys (state, action, next_state) and values as rewards
        :param V: Optimal value function
        :return: Optimal deterministic policy as a dictionary {state: best_action}
        """
        policy = {}
        for st in self.gr_states:
            best_action = max(self.gr_actions, key=lambda act:
            rewards[st][act] + self.gamma * sum(
                self.gr_transition[st][act][st_prime] * V[st_prime]
                for st_prime in self.gr_transition[st][act].keys()
            ))
            policy[st] = {}
            for act in self.gr_actions:
                if act == best_action:
                    policy[st][act] = 1
                else:
                    policy[st][act] = 0
        return policy

    def value_iteration(self, rewards, theta=1e-6):
        """
        Performs Value Iteration to compute the optimal state values.

        :param rewards: Dictionary with keys (state, action) and values as rewards
        :param theta: Convergence threshold
        :return: Optimal value function and optimal policy
        """
        V = {st: 0 for st in self.gr_states}  # Initialize value function to zero

        while True:
            delta = 0
            for st in self.gr_states:
                v = V[st]
                V[st] = max(
                    rewards[st][act] + self.gamma * sum(
                        self.gr_transition[st][act][st_prime] * V[st_prime]
                        for st_prime in self.gr_transition[st][act].keys())
                    for act in self.gr_actions
                )
                delta = max(delta, abs(v - V[st]))
            if delta < theta:
                break

        return V, self.extract_policy(rewards, V)

    def get_states_transition(self, type):
        trans = {}
        for st in self.gr_states:
            trans[st] = {}
            for st_prime in self.gr_states:
                trans[st][st_prime] = 0
                for act in self.gr_actions:
                    if type == 'n':
                        # print(st, act, st_prime)
                        if st_prime in self.gr_transition[st][act].keys():
                            trans[st][st_prime] += self.gr_transition[st][act][st_prime] * self.opt_policy_n[st][act]
                        else:
                            trans[st][st_prime] += 0
                    elif type == 'a':
                        if st_prime in self.gr_transition[st][act].keys():
                            trans[st][st_prime] += self.gr_transition[st][act][st_prime] * self.opt_policy_a[st][act]
                        else:
                            trans[st][st_prime] += 0
                    else:
                        raise ValueError('Invalid type parameter.')
        return trans
