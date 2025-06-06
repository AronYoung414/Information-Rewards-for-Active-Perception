import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from collections import deque
import random

from product_pomdp import prod_pomdp
from information_rewards import InformationRewards

# prod_pomdp = prod_pomdp()
info_rewards = InformationRewards()


class ActorNetwork(nn.Module):
    """Actor network for agent_2's policy"""

    def __init__(self, input_size, hidden_size=128, output_size=3):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


class CriticNetwork(nn.Module):
    """Critic network for value function estimation"""

    def __init__(self, input_size, hidden_size=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent2ActorCritic:
    """Deep Actor-Critic agent for agent_2"""

    def __init__(self, env, T=10, lr_actor=0.0003, lr_critic=0.0005,
                 gamma=0.99, hidden_size=128, window_size=1000):
        self.env = env
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The size of state
        # Total: 1 + 1 + 1 + 1 = 4 features
        self.state_size = 4

        # the length of observation
        self.observation_length = T

        # Initialize networks - Actor uses observation_length, Critic uses state_size
        self.actor = ActorNetwork(self.observation_length, hidden_size).to(self.device)
        self.critic = CriticNetwork(self.state_size, hidden_size).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Experience buffer
        self.episode_states = []
        self.episode_ss = []  # The list of indices of states
        self.episode_actions = []
        self.episode_as = []  # The list of indices of actions
        self.episode_rewards = []
        self.episode_log_probs = []
        self.episode_values = []
        self.episode_obs = []  # The list of observations

    def encode_state(self, state):
        if state in self.env.sink_states:
            if state == 'sink1':  # Encode the sink states 1
                state_vector = np.array([
                    -1 / 6,  # 1 features
                    -1 / 6,  # 1 features
                    0,  # 1 features
                    0  # 1 feature
                ])
            elif state == 'sink2':  # Encode the sink states 2
                state_vector = np.array([
                    -2 / 6,  # 1 features
                    -2 / 6,  # 1 features
                    0,  # 1 features
                    0  # 1 feature
                ])
            else:  # Encode the sink states 3
                state_vector = np.array([
                    -3 / 6,  # 1 features
                    -3 / 6,  # 1 features
                    0,  # 1 features
                    0  # 1 feature
                ])
        else:
            """Encode the current state into a feature vector"""
            agent1_pos = state[0][0]
            agent2_pos = state[0][1]
            type = state[0][2]
            auto_st = state[1]
            # Normalize positions to [0, 1]
            agent1_norm = float(agent1_pos) / 6  # There are 6 states in the graph
            agent2_norm = float(agent2_pos) / 6  # There are 6 states in the graph

            # # Relative position
            # relative_pos = agent1_norm - agent2_norm

            # Concatenate all features
            state_vector = np.array([
                agent1_norm,  # 1 features
                agent2_norm,  # 1 features
                type,  # 1 features
                auto_st  # 1 feature
            ])

        return torch.FloatTensor(state_vector).to(self.device)

    def encode_observation(self, obs):
        """Encode a single observation into a numerical value"""
        if obs is None:
            return 0.0

        # Map your specific observations to numerical values
        obs_mapping = {
            '0': 0.0,
            '1': 1.0,
            '2': 2.0,
            '3': 3.0,
            '4': 4.0,
            '5': 5.0,
            'n': 6.0  # 'n' mapped to 6.0
        }

        return obs_mapping.get(obs, 0.0)  # Default to 0.0 if observation not found

    def prepare_observation_sequence(self, episode_obs, max_length=None):
        """Prepare observation sequence for the actor network"""
        if max_length is None:
            max_length = self.observation_length

        # Encode observations to numerical values
        encoded_obs = [self.encode_observation(obs) for obs in episode_obs]

        # If we don't have enough observations, pad with zeros
        if len(encoded_obs) < max_length:
            # Pad with zeros at the beginning
            padded_obs = [0.0] * (max_length - len(encoded_obs)) + encoded_obs
        else:
            # Take the last max_length observations
            padded_obs = encoded_obs[-max_length:]

        return torch.FloatTensor(padded_obs).to(self.device)

    def select_action(self, obs_sequence):
        """Select an action using the actor network"""
        action_probs = self.actor(obs_sequence)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def update_networks(self):
        """Update actor and critic networks using collected episode data"""
        if len(self.episode_rewards) == 0:
            return None, None

        # Convert lists to tensors
        # states = torch.stack(self.episode_states)
        # actions = torch.LongTensor(self.episode_actions).to(self.device)
        rewards = torch.FloatTensor(self.episode_rewards).to(self.device)
        log_probs = torch.stack(self.episode_log_probs)
        values = torch.stack(self.episode_values).squeeze()

        # Compute returns (discounted cumulative rewards)
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize returns
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / returns.std()

        # Compute advantages
        advantages = returns - values.detach()

        # Actor loss (policy gradient with baseline)
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss (MSE between predicted values and returns)
        critic_loss = F.mse_loss(values, returns)

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        # Clear episode data
        self.episode_states.clear()
        self.episode_ss.clear()
        self.episode_actions.clear()
        self.episode_as.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
        self.episode_values.clear()
        self.episode_obs.clear()

        return actor_loss.item(), critic_loss.item()

    def train_episode(self, max_steps=20, alpha=1):
        """Train for one episode"""
        # obs, _ = self.env.reset()
        total_reward = 0
        total_entropy = 0
        total_probs = 0

        state = random.choices(self.env.initial_states, self.env.initial_dist_sampling, k=1)[0]
        self.episode_states.append(state)
        s = self.env.states.index(state)
        self.episode_ss.append(s)

        for step in range(max_steps):
            # Encode state for critic
            state_tensor = self.encode_state(state)

            # Get value estimate from critic
            value = self.critic(state_tensor)

            # Prepare observation sequence for actor
            obs_sequence = self.prepare_observation_sequence(self.episode_obs)

            # Select action for agent_2 using observation sequence
            agent2_action, log_prob = self.select_action(obs_sequence)
            act = self.env.actions[agent2_action]
            self.episode_as.append(agent2_action)

            # Get observations
            obs = self.env.observation_function_sampler(state, act)
            self.episode_obs.append(obs)

            # Take step in environment
            state = self.env.next_state_sampler(state, act)
            s = self.env.states.index(state)

            # Compute entropy-based reward for agent_2
            entropy_diff, prob_diff = info_rewards.reward_function(self.episode_obs, self.episode_as)
            reward = entropy_diff - alpha * prob_diff  # alpha is the weighting parameter

            # Store experience
            self.episode_states.append(state)
            self.episode_ss.append(s)
            self.episode_actions.append(act)
            self.episode_rewards.append(reward)
            self.episode_log_probs.append(log_prob)
            self.episode_values.append(value)

            total_reward += reward
            total_entropy += entropy_diff
            total_probs += prob_diff

            # Check if agent_1 reached goal (episode termination)
            if state in self.env.sink_states:
                break

        # Final step handling - but don't add to training data since no action was chosen by policy
        act = 'e'  # Ending action
        agent2_action = self.env.actions.index(act)
        final_state = self.env.next_state_sampler(state, act)
        final_state_tensor = self.encode_state(final_state)
        s_final = self.env.states.index(final_state)
        obs_final = self.env.observation_function_sampler(final_state, act)
        final_log_prob = torch.tensor(0).to(self.device)  # With probability 1 to select the ending action.

        self.episode_as.append(agent2_action)
        self.episode_obs.append(obs_final)

        # Compute final reward but don't include in training
        entropy_diff, prob_diff = info_rewards.reward_function(self.episode_obs, self.episode_as)
        final_reward = entropy_diff - alpha * prob_diff

        final_value = self.critic(final_state_tensor)

        self.episode_states.append(final_state)
        self.episode_ss.append(s_final)
        self.episode_rewards.append(final_reward)
        self.episode_values.append(final_value)
        self.episode_log_probs.append(final_log_prob)

        total_reward += final_reward
        total_entropy += entropy_diff
        total_probs += prob_diff

        # Update networks at end of episode
        actor_loss, critic_loss = self.update_networks()

        return total_reward, total_entropy, total_probs, step + 1, actor_loss, critic_loss


def train_agent2_actor_critic(env, num_episodes=1000, window_size=50):
    """Train agent_2 using actor-critic algorithm"""

    # Initialize agent
    agent2 = Agent2ActorCritic(env)

    # Training metrics
    episode_rewards = []
    episode_entropies = []
    episode_probs = []
    episode_lengths = []
    actor_losses = []
    critic_losses = []

    # For tracking moving average
    recent_rewards = deque(maxlen=window_size)

    print("Starting training...")
    print("Episode | Reward | MA Reward | Length | Actor Loss | Critic Loss | Entropy | Probs")
    print("-" * 85)

    for episode in range(num_episodes):
        # Train one episode
        total_reward, total_entropy, total_probs, episode_length, actor_loss, critic_loss = agent2.train_episode()

        # Store metrics
        episode_rewards.append(total_reward)
        episode_entropies.append(-total_entropy)
        episode_probs.append(-total_probs)
        episode_lengths.append(episode_length)
        recent_rewards.append(total_reward)
        if actor_loss is not None:
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        # Calculate moving average reward
        moving_avg_reward = sum(recent_rewards) / len(recent_rewards)

        # Print progress
        if episode % 50 == 0:
            current_entropy = episode_entropies[-1]
            current_probs = episode_probs[-1]
            actor_loss_str = f"{actor_loss:10.4f}" if actor_loss is not None else "    None"
            critic_loss_str = f"{critic_loss:11.4f}" if critic_loss is not None else "     None"
            print(f"{episode:7d} | {total_reward:6.3f} | {moving_avg_reward:9.3f} | {episode_length:6d} | "
                  f"{actor_loss_str} | {critic_loss_str} | {current_entropy:7.4f} | {current_probs:7.4f}")

        # Evaluate periodically
        # if episode % eval_interval == 0 and episode > 0:
        #     mean_reward, std_reward, mean_entropy, std_entropy = agent2.evaluate()
        #     print(f"\nEvaluation after {episode} episodes:")
        #     print(f"  Average reward: {mean_reward:.3f} ± {std_reward:.3f}")
        #     print(f"  Average entropy: {mean_entropy:.3f} ± {std_entropy:.3f}")
        #     print("-" * 85)

    return agent2, episode_rewards, episode_entropies, episode_probs, actor_losses, critic_losses


def visualize_training_results(episode_rewards, episode_entropies, episode_probs, actor_losses, critic_losses,
                               window=50):
    """Visualize training progress with moving averages"""
    import matplotlib.pyplot as plt
    import numpy as np

    # Compute moving averages
    def moving_average(data, window_size):
        if len(data) < window_size:
            return data

        cumsum_vec = np.cumsum(np.insert(data, 0, 0))
        ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

        # Pad the beginning to match original length
        return np.concatenate([data[:window_size - 1], ma_vec])

    # Calculate moving averages
    rewards_ma = moving_average(episode_rewards, window)
    entropy_ma = moving_average(episode_entropies, window)
    probs_ma = moving_average(episode_probs, window)

    if actor_losses and len(actor_losses) > window:
        actor_losses_ma = moving_average(actor_losses, window)
        critic_losses_ma = moving_average(critic_losses, window)
    else:
        actor_losses_ma = actor_losses
        critic_losses_ma = critic_losses

    # Fix: Create subplots properly
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax1, ax2, ax5 = axes[0, 0], axes[0, 1], axes[0, 2]
    ax3, ax4 = axes[1, 0], axes[1, 1]
    axes[1, 2].axis('off')  # Hide the unused subplot

    # Episode rewards
    ax1.plot(episode_rewards, alpha=0.3, color='blue', label='Raw')
    ax1.plot(rewards_ma, linewidth=2, color='blue', label=f'Moving Avg (window={window})')
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    ax1.legend()

    # Episode entropies
    ax2.plot(episode_entropies, alpha=0.3, color='green', label='Raw')
    ax2.plot(entropy_ma, linewidth=2, color='green', label=f'Moving Avg (window={window})')
    ax2.set_title('Episode Entropies')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Entropy')  # Fixed label
    ax2.grid(True)
    ax2.legend()

    # Episode probs of completing tasks
    ax5.plot(episode_probs, alpha=0.3, color='black', label='Raw')
    ax5.plot(probs_ma, linewidth=2, color='black', label=f'Moving Avg (window={window})')
    ax5.set_title('Episode Probabilities of Completing Tasks')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Probability')  # Fixed label
    ax5.grid(True)
    ax5.legend()

    # Actor losses
    if actor_losses:
        ax3.plot(actor_losses, alpha=0.3, color='red', label='Raw')
        ax3.plot(actor_losses_ma, linewidth=2, color='red', label=f'Moving Avg (window={window})')
        ax3.set_title('Actor Loss')
        ax3.set_xlabel('Update')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
        ax3.legend()

    # Critic losses
    if critic_losses:
        ax4.plot(critic_losses, alpha=0.3, color='purple', label='Raw')
        ax4.plot(critic_losses_ma, linewidth=2, color='purple', label=f'Moving Avg (window={window})')
        ax4.set_title('Critic Loss')
        ax4.set_xlabel('Update')
        ax4.set_ylabel('Loss')
        ax4.grid(True)
        ax4.legend()

    plt.tight_layout()
    plt.show()


def save_data(rewards, entropies, probs, actor_losses, critic_losses, ex_num=1):
    with open(f'./ac_data/rewards_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(rewards, pkl_wb_obj)

    with open(f'./ac_data/entropies_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(entropies, pkl_wb_obj)

    with open(f'./ac_data/probs_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(probs, pkl_wb_obj)

    with open(f'./ac_data/actor_losses_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(actor_losses, pkl_wb_obj)

    with open(f'./ac_data/critic_losses_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(critic_losses, pkl_wb_obj)

    return 0


def main():
    # Create environment
    env = prod_pomdp()

    # Compute optimal policy for agent_1
    # print("Computing agent_1's optimal policy...")
    # _, agent1_policy = value_iteration(env)

    # Train agent_2
    print("Training agent_2 with entropy minimization rewards...")
    agent2, rewards, entropies, probs, actor_losses, critic_losses = train_agent2_actor_critic(
        env, num_episodes=10000)

    # Save the data
    save_data(rewards, entropies, probs, actor_losses, critic_losses, ex_num=1)

    # Final evaluation
    # print("\nFinal evaluation:")
    # mean_reward, std_reward, mean_entropy, std_entropy = agent2.evaluate(num_episodes=50)
    # print(f"Average reward: {mean_reward:.3f} ± {std_reward:.3f}")
    # print(f"Average entropy: {mean_entropy:.3f} ± {std_entropy:.3f}")

    # Visualize results
    print("Generating training plots...")
    visualize_training_results(rewards, entropies, probs, actor_losses, critic_losses, window=50)

    # Save trained model
    torch.save({
        'actor_state_dict': agent2.actor.state_dict(),
        'critic_state_dict': agent2.critic.state_dict(),
        'actor_optimizer_state_dict': agent2.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent2.critic_optimizer.state_dict(),
    }, 'agent2_actor_critic_model.pth')

    print("Training completed and model saved!")

    return agent2


if __name__ == "__main__":
    main()
