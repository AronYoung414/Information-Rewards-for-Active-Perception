import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

from product_pomdp import prod_pomdp
# from information_rewards import particle_filter
from particle_filter_estimation import particle_filter


class ActorNetwork(nn.Module):
    """Improved Actor network with better initialization and regularization"""

    def __init__(self, input_size, hidden_size=128, output_size=3, dropout_rate=0.2):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

        # Better weight initialization
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

        # Smaller initialization for output layer to prevent extreme probabilities
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.softmax(x)


class CriticNetwork(nn.Module):
    """Improved Critic network with better architecture"""

    def __init__(self, input_size, hidden_size=128, dropout_rate=0.2):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # Additional layer
        self.fc4 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class Agent2ActorCritic:
    """Improved Deep Actor-Critic agent with variance reduction techniques"""

    def __init__(self, env, T=10, lr_actor=0.0001, lr_critic=0.0003,
                 gamma=0.95, hidden_size=128, window_size=1000,
                 entropy_coeff=0.01, value_loss_coeff=0.5, max_grad_norm=0.5,
                 batch_size=32, use_gae=True, gae_lambda=0.95):
        self.env = env
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_size = 6  # 6 features
        self.observation_length = T

        # Initialize networks with improved architecture
        self.actor = ActorNetwork(self.observation_length, hidden_size).to(self.device)
        self.critic = CriticNetwork(self.state_size, hidden_size).to(self.device)

        # Use different optimizers with weight decay
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr_actor, weight_decay=1e-4)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr_critic, weight_decay=1e-4)

        # Learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.95)

        # Experience buffer
        self.reset_episode_buffers()

        # Experience replay buffer for better sample efficiency
        self.replay_buffer = deque(maxlen=10000)

        # Running statistics for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

        # Target networks for stability (optional)
        self.use_target_network = True
        if self.use_target_network:
            self.target_critic = CriticNetwork(self.state_size, hidden_size).to(self.device)
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.target_update_freq = 100
            self.update_count = 0

    def reset_episode_buffers(self):
        self.episode_states = []
        self.episode_ss = []
        self.episode_actions = []
        self.episode_as = []
        self.episode_rewards = []
        self.episode_log_probs = []
        self.episode_values = []
        self.episode_obs = []
        self.episode_dones = []

    def update_reward_stats(self, reward):
        """Update running statistics for reward normalization"""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_std = np.sqrt(
            max(1e-8, (self.reward_count - 1) * self.reward_std ** 2 + delta * delta2) / self.reward_count)

    def normalize_reward(self, reward):
        """Normalize reward using running statistics"""
        return (reward - self.reward_mean) / max(self.reward_std, 1e-8)

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_value_est = next_value
            else:
                next_non_terminal = 1.0 - dones[i]
                next_value_est = values[i + 1]

            delta = rewards[i] + self.gamma * next_value_est * next_non_terminal - values[i]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)

        return advantages

    def encode_state(self, state):
        if state in self.env.sink_states:
            if state == 'sink1':
                state_vector = np.array([-1, -1, 0, 0, 0, 0])  # [agent1_x, agent1_y, agent2_x, agent2_y, type, auto_st]
            elif state == 'sink2':
                state_vector = np.array([0, 0, -1, -1, 0, 0])
            else:
                state_vector = np.array([-1, -1, -1, -1, 0, 0])
        else:
            agent1_pos = state[0][0]
            agent2_pos = state[0][1]
            type = state[0][2]
            auto_st = state[1]
            agent1_norm = np.array(agent1_pos) / np.array([self.env.width - 1, self.env.height - 1])
            agent2_norm = np.array(agent2_pos) / np.array([self.env.width - 1, self.env.height - 1])

            # Flatten into a single 1D array
            state_vector = np.concatenate([agent1_norm, agent2_norm, [type, auto_st]])

        return torch.FloatTensor(state_vector).to(self.device)

    def encode_observation(self, obs):
        """Encode observation consistently"""
        if obs is None:
            return np.array([0.0, 0.0])  # Return array instead of scalar

        obs_mapping = {}
        for obs_key in self.env.observations:
            if obs_key == 'n':
                obs_mapping[obs_key] = np.array([-1.0, -1.0])
            else:
                obs_mapping[obs_key] = np.array(obs_key) / np.array([self.env.width - 1, self.env.height - 1])

        return obs_mapping.get(obs, np.array([0.0, 0.0]))  # Return array instead of scalar

    def prepare_observation_sequence(self, episode_obs, max_length=None):
        """Prepare observation sequence for neural network input"""
        if max_length is None:
            max_length = self.observation_length

        encoded_obs = []
        for obs in episode_obs:
            encoded_obs_item = self.encode_observation(obs)
            encoded_obs.append(encoded_obs_item)

        # Convert everything to a flat list of scalars
        flattened_obs = []
        for encoded_obs_item in encoded_obs:
            if isinstance(encoded_obs_item, np.ndarray):
                flattened_obs.extend(encoded_obs_item.flatten().tolist())
            else:
                # If it's a scalar (like 0.0), convert to a list
                flattened_obs.append(float(encoded_obs_item))

        # Pad or truncate to match expected length
        if len(flattened_obs) < max_length:
            padded_obs = [0.0] * (max_length - len(flattened_obs)) + flattened_obs
        else:
            padded_obs = flattened_obs[-max_length:]

        return torch.FloatTensor(padded_obs).to(self.device)

    def select_action(self, obs_sequence, training=True):
        """Select action with improved exploration"""
        action_probs = self.actor(obs_sequence)

        # Add small epsilon for numerical stability
        action_probs = action_probs + 1e-8
        action_probs = action_probs / action_probs.sum()

        dist = torch.distributions.Categorical(action_probs)

        if training:
            action = dist.sample()
        else:
            action = torch.argmax(action_probs)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.item(), log_prob, entropy

    def update_networks(self):
        """Improved network updates with multiple techniques"""
        if len(self.episode_rewards) == 0:
            return None, None, None

        # Convert to tensors
        rewards = torch.FloatTensor(self.episode_rewards).to(self.device)
        log_probs = torch.stack(self.episode_log_probs)
        values = torch.stack(self.episode_values).squeeze()
        dones = torch.FloatTensor(self.episode_dones).to(self.device)

        # Normalize rewards
        normalized_rewards = torch.FloatTensor([
            self.normalize_reward(r.item()) for r in rewards
        ]).to(self.device)

        # Update reward statistics
        for reward in rewards:
            self.update_reward_stats(reward.item())

        # Compute next value for GAE (assuming episode ends)
        next_value = torch.tensor(0.0).to(self.device)

        if self.use_gae:
            # Use GAE for advantage computation
            advantages = self.compute_gae(
                normalized_rewards.cpu().numpy(),
                values.detach().cpu().numpy(),
                dones.cpu().numpy(),
                next_value.item()
            )
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = advantages + values.detach()
        else:
            # Standard return computation
            returns = []
            G = next_value
            for reward in reversed(normalized_rewards):
                G = reward + self.gamma * G
                returns.insert(0, G)
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = returns - values.detach()

        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute entropy for regularization
        entropies = []
        for obs in self.episode_obs[:-1]:  # Exclude final observation
            obs_seq = self.prepare_observation_sequence([obs])
            action_probs = self.actor(obs_seq)
            dist = torch.distributions.Categorical(action_probs)
            entropies.append(dist.entropy())

        if entropies:
            entropy_loss = torch.stack(entropies).mean()
        else:
            entropy_loss = torch.tensor(0.0).to(self.device)

        # Actor loss with entropy regularization
        policy_loss = -(log_probs * advantages).mean()
        actor_loss = policy_loss - self.entropy_coeff * entropy_loss

        # Critic loss with clipping
        value_loss = F.mse_loss(values, returns)

        # Multiple gradient updates for better convergence
        num_updates = max(1, len(self.episode_rewards) // self.batch_size)

        total_actor_loss = 0
        total_critic_loss = 0

        for _ in range(num_updates):
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += value_loss.item()

        # Update learning rates
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # Update target network
        if self.use_target_network:
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_critic.load_state_dict(self.critic.state_dict())

        # Store experience in replay buffer
        episode_data = {
            'states': self.episode_states.copy(),
            'actions': self.episode_as.copy(),
            'rewards': self.episode_rewards.copy(),
            'observations': self.episode_obs.copy()
        }
        self.replay_buffer.append(episode_data)

        # Clear episode data
        self.reset_episode_buffers()

        return total_actor_loss / num_updates, total_critic_loss / num_updates, entropy_loss.item()

    # def train_episode(self, max_steps=20, alpha=1):
    #     """Train for one episode with improved stability"""
    #     total_reward = 0
    #     total_entropy = 0
    #     total_probs = 0
    #
    #     state = random.choices(self.env.initial_states, self.env.initial_dist_sampling, k=1)[0]
    #     self.episode_states.append(state)
    #     s = self.env.states.index(state)
    #     self.episode_ss.append(s)
    #
    #     for step in range(max_steps):
    #         state_tensor = self.encode_state(state)
    #         value = self.critic(state_tensor)
    #
    #         obs_sequence = self.prepare_observation_sequence(self.episode_obs)
    #         agent2_action, log_prob, entropy = self.select_action(obs_sequence, training=True)
    #         act = self.env.actions[agent2_action]
    #         self.episode_as.append(agent2_action)
    #
    #         obs = self.env.observation_function_sampler(state, act)
    #         self.episode_obs.append(obs)
    #
    #         state = self.env.next_state_sampler(state, act)
    #         s = self.env.states.index(state)
    #
    #         entropy_diff, prob_diff = info_rewards.reward_function(self.episode_obs, self.episode_as)
    #         reward = entropy_diff - alpha * prob_diff
    #
    #         # Check if episode should terminate
    #         done = state in self.env.sink_states
    #
    #         self.episode_states.append(state)
    #         self.episode_ss.append(s)
    #         self.episode_actions.append(act)
    #         self.episode_rewards.append(reward)
    #         self.episode_log_probs.append(log_prob)
    #         self.episode_values.append(value)
    #         self.episode_dones.append(done)
    #
    #         total_reward += reward
    #         total_entropy += entropy_diff
    #         total_probs += prob_diff
    #
    #         if done:
    #             break
    #
    #     # Final step handling
    #     act = 'e'
    #     agent2_action = self.env.actions.index(act)
    #     final_state = self.env.next_state_sampler(state, act)
    #     final_state_tensor = self.encode_state(final_state)
    #     s_final = self.env.states.index(final_state)
    #     obs_final = self.env.observation_function_sampler(final_state, act)
    #     final_log_prob = torch.tensor(0.0).to(self.device)
    #
    #     self.episode_as.append(agent2_action)
    #     self.episode_obs.append(obs_final)
    #
    #     entropy_diff, prob_diff = info_rewards.reward_function(self.episode_obs, self.episode_as)
    #     final_reward = entropy_diff - alpha * prob_diff
    #
    #     final_value = self.critic(final_state_tensor)
    #
    #     self.episode_states.append(final_state)
    #     self.episode_ss.append(s_final)
    #     self.episode_rewards.append(final_reward)
    #     self.episode_values.append(final_value)
    #     self.episode_log_probs.append(final_log_prob)
    #     self.episode_dones.append(True)
    #
    #     total_reward += final_reward
    #     total_entropy += entropy_diff
    #     total_probs += prob_diff
    #
    #     # Update networks
    #     actor_loss, critic_loss, entropy_loss = self.update_networks()
    #
    #     return total_reward, total_entropy, total_probs, step + 1, actor_loss, critic_loss

    def train_episode(self, max_steps=20, alpha=0.5):
        """Train for one episode with improved stability"""
        # Initialize total reward, entropy and reaching probability
        total_reward = 0
        total_entropy = 0
        total_probs = 0
        # Initialize the value of current entropy and reaching probability
        entropy_now = 0
        p_wT1_now = 0

        state = random.choices(self.env.initial_states, self.env.initial_dist_sampling, k=1)[0]
        self.episode_states.append(state)
        s = self.env.states.index(state)
        self.episode_ss.append(s)

        pf = particle_filter(self.env, state, num_particles=1000)

        for step in range(max_steps - 1):
            state_tensor = self.encode_state(state)
            value = self.critic(state_tensor)

            obs_sequence = self.prepare_observation_sequence(self.episode_obs)
            agent2_action, log_prob, entropy = self.select_action(obs_sequence, training=True)
            act = self.env.actions[agent2_action]
            self.episode_as.append(agent2_action)

            obs = self.env.observation_function_sampler(state, act)
            self.episode_obs.append(obs)

            state = self.env.next_state_sampler(state, act)
            s = self.env.states.index(state)
            # Calculate entropy and probability difference
            entropy_before = entropy_now
            p_wT1_before = p_wT1_now
            entropy_now, p_wT1_now = pf.calculate_entropy(obs, act)
            # entropy difference
            entropy_diff = entropy_before - entropy_now
            # probability difference
            prob_diff = p_wT1_before - p_wT1_now
            # obtain rewards
            reward = entropy_diff - alpha * prob_diff

            # Check if episode should terminate
            #done = state in self.env.sink_states

            self.episode_states.append(state)
            self.episode_ss.append(s)
            self.episode_actions.append(act)
            self.episode_rewards.append(reward)
            self.episode_log_probs.append(log_prob)
            self.episode_values.append(value)
            self.episode_dones.append(False)

            total_reward += reward
            total_entropy += entropy_diff
            total_probs += prob_diff

            #if done:
               # break

        # Final step handling
        final_act = 'e'
        agent2_action = self.env.actions.index(final_act)
        final_state = self.env.next_state_sampler(state, final_act)
        final_state_tensor = self.encode_state(final_state)
        final_s = self.env.states.index(final_state)
        final_obs = self.env.observation_function_sampler(final_state, final_act)
        final_log_prob = torch.tensor(0.0).to(self.device)

        self.episode_as.append(agent2_action)
        self.episode_obs.append(final_obs)

        entropy_before = entropy_now
        p_wT1_before = p_wT1_now
        entropy_now, p_wT1_now = pf.calculate_entropy(final_obs, final_act)
        # entropy difference
        entropy_diff = entropy_before - entropy_now
        # probability difference
        prob_diff = p_wT1_before - p_wT1_now
        # obtain rewards
        final_reward = entropy_diff - alpha * prob_diff

        final_value = self.critic(final_state_tensor)

        self.episode_states.append(final_state)
        self.episode_ss.append(final_s)
        self.episode_rewards.append(final_reward)
        self.episode_values.append(final_value)
        self.episode_log_probs.append(final_log_prob)
        self.episode_dones.append(True)

        #print(self.episode_obs)

        total_reward += final_reward
        total_entropy += entropy_diff
        total_probs += prob_diff
        #print(total_entropy)

        # Update networks
        actor_loss, critic_loss, entropy_loss = self.update_networks()


        return total_reward, total_entropy, total_probs, step + 1, actor_loss, critic_loss


def train_agent2_actor_critic(env, num_episodes=1000, window_size=50):
    """Train agent_2 using improved actor-critic algorithm"""

    # Initialize agent with better hyperparameters
    agent2 = Agent2ActorCritic(
        env,
        lr_actor=0.0005,  # Lower learning rate
        lr_critic=0.006,  # Lower learning rate
        gamma=0.95,  # Slightly lower discount
        entropy_coeff=0.01,  # Entropy regularization
        use_gae=True,  # Use GAE
        gae_lambda=0.95
    )

    # Training metrics
    episode_rewards = []
    episode_entropies = []
    episode_probs = []
    episode_lengths = []
    actor_losses = []
    critic_losses = []

    # For tracking moving average
    recent_rewards = deque(maxlen=window_size)

    print("Starting improved training...")
    print("Episode | Reward | MA Reward | Length | Actor Loss | Critic Loss | Entropy | Probs")
    print("-" * 85)

    for episode in range(num_episodes):
        total_reward, total_entropy, total_probs, episode_length, actor_loss, critic_loss = agent2.train_episode()

        episode_rewards.append(total_reward)
        episode_entropies.append(-total_entropy)
        episode_probs.append(-total_probs)
        episode_lengths.append(episode_length)
        recent_rewards.append(total_reward)

        if actor_loss is not None:
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        moving_avg_reward = sum(recent_rewards) / len(recent_rewards)

        if episode % 50 == 0:
            current_entropy = episode_entropies[-1]
            current_probs = episode_probs[-1]
            actor_loss_str = f"{actor_loss:10.4f}" if actor_loss is not None else "    None"
            critic_loss_str = f"{critic_loss:11.4f}" if critic_loss is not None else "     None"
            print(f"{episode:7d} | {total_reward:6.3f} | {moving_avg_reward:9.3f} | {episode_length:6d} | "
                  f"{actor_loss_str} | {critic_loss_str} | {current_entropy:7.4f} | {current_probs:7.4f}")

    return agent2, episode_rewards, episode_entropies, episode_probs, actor_losses, critic_losses


def visualize_training_results(episode_rewards, episode_entropies, episode_probs, actor_losses, critic_losses,
                               ex_num, window=50):
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
    plt.savefig(f'./ac_data/result_{ex_num}.png')
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
    ex_num = 9

    # Compute optimal policy for agent_1
    # print("Computing agent_1's optimal policy...")
    # _, agent1_policy = value_iteration(env)

    # Train agent_2
    print("Training agent_2 with entropy minimization rewards...")
    agent2, rewards, entropies, probs, actor_losses, critic_losses = train_agent2_actor_critic(
        env, num_episodes=10000)

    # Save the data
    save_data(rewards, entropies, probs, actor_losses, critic_losses, ex_num)

    # Final evaluation
    # print("\nFinal evaluation:")
    # mean_reward, std_reward, mean_entropy, std_entropy = agent2.evaluate(num_episodes=50)
    # print(f"Average reward: {mean_reward:.3f} ± {std_reward:.3f}")
    # print(f"Average entropy: {mean_entropy:.3f} ± {std_entropy:.3f}")

    # Visualize results
    print("Generating training plots...")
    visualize_training_results(rewards, entropies, probs, actor_losses, critic_losses, ex_num, window=50)

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
