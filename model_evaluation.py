from collections import defaultdict

# You would call this after training your model
from actor_critic_reduce_variance import *


def load_trained_agent_and_generate_trajectories(model_path, env, num_trajectories=10, max_steps=15,
                                                 alpha=0, seed=None, save_trajectories=False,
                                                 output_file=None, verbose=True):
    """
    Load a trained Actor-Critic model and generate state trajectories.

    Args:
        model_path (str): Path to the saved .pth model file
        env: The POMDP environment instance
        num_trajectories (int): Number of trajectories to generate
        max_steps (int): Maximum steps per trajectory (should match training T)
        alpha (float): Alpha parameter for reward calculation
        seed (int): Random seed for reproducibility
        save_trajectories (bool): Whether to save trajectories to file
        output_file (str): File path to save trajectories (if save_trajectories=True)
        verbose (bool): Whether to print trajectory details

    Returns:
        trajectories (list): List of trajectory dictionaries containing:
            - states: sequence of states
            - actions: sequence of actions
            - observations: sequence of observations
            - rewards: sequence of rewards
            - total_reward: total trajectory reward
            - total_entropy: total entropy change
            - total_probs: total probability change
    """

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Create agent instance with same parameters as training
    agent = Agent2ActorCritic(
        env, T=max_steps,
        lr_actor=0.0001,
        lr_critic=0.0003,
        gamma=0.95,
        entropy_coeff=0.01,
        use_gae=True,
        gae_lambda=0.95
    )

    # Load the trained model
    checkpoint = torch.load(model_path, map_location=agent.device)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])

    # Set networks to evaluation mode
    agent.actor.eval()
    agent.critic.eval()

    trajectories = []

    if verbose:
        print(f"Generating {num_trajectories} trajectories...")
        print("Traj | Total Reward | Total Entropy | Total Probs | Length | Final State")
        print("-" * 75)

    for traj_idx in range(num_trajectories):
        # Initialize trajectory data
        trajectory = {
            'states': [],
            'actions': [],
            'observations': [],
            'rewards': [],
            'state_encodings': [],
            'action_probs': []
        }

        # Initialize episode
        total_reward = 0
        total_entropy = 0
        total_probs = 0
        entropy_now = 0
        p_wT1_now = 0

        # Sample initial state
        state = random.choices(env.initial_states, env.initial_dist_sampling, k=1)[0]
        trajectory['states'].append(state)

        # Initialize particle filter (reusing from your code)
        pf = particle_filter(env, state, num_particles=1000)

        episode_obs = []

        # Generate trajectory
        for step in range(max_steps - 1):
            # Prepare observation sequence for action selection
            obs_sequence = agent.prepare_observation_sequence(episode_obs, max_length=max_steps)

            # Select action using trained actor (no training mode)
            with torch.no_grad():
                action_probs = agent.actor(obs_sequence)
                action_probs = action_probs + 1e-8  # Numerical stability
                action_probs = action_probs / action_probs.sum()

                # Use deterministic policy for evaluation (take most probable action)
                # or stochastic for exploration - you can choose
                action_idx = torch.argmax(action_probs).item()  # Deterministic
                # action_idx = torch.multinomial(action_probs, 1).item()  # Stochastic

            act = env.actions[action_idx]
            trajectory['actions'].append(act)
            trajectory['action_probs'].append(action_probs.cpu().numpy())

            # Get observation and next state
            obs = env.observation_function_sampler(state, act)
            episode_obs.append(obs)
            trajectory['observations'].append(obs)

            next_state = env.next_state_sampler(state, act)

            # Calculate entropy and probability changes
            entropy_before = entropy_now
            p_wT1_before = p_wT1_now
            entropy_now, p_wT1_now = pf.calculate_entropy(obs, act)

            # Calculate reward
            entropy_diff = entropy_before - entropy_now
            prob_diff = p_wT1_before - p_wT1_now
            reward = entropy_diff - alpha * prob_diff

            trajectory['rewards'].append(reward)
            total_reward += reward
            total_entropy += entropy_diff
            total_probs += prob_diff

            # Update state
            state = next_state
            trajectory['states'].append(state)

            # Store state encoding for analysis
            state_encoding = agent.encode_state(state).cpu().numpy()
            trajectory['state_encodings'].append(state_encoding)

        # Final step
        final_act = 'e'
        final_action_idx = env.actions.index(final_act)
        trajectory['actions'].append(final_act)

        final_state = env.next_state_sampler(state, final_act)
        final_obs = env.observation_function_sampler(final_state, final_act)
        episode_obs.append(final_obs)
        trajectory['observations'].append(final_obs)

        # Final entropy and probability calculation
        entropy_before = entropy_now
        p_wT1_before = p_wT1_now
        entropy_now, p_wT1_now = pf.calculate_entropy(final_obs, final_act)

        entropy_diff = entropy_before - entropy_now
        prob_diff = p_wT1_before - p_wT1_now
        final_reward = entropy_diff - alpha * prob_diff

        trajectory['rewards'].append(final_reward)
        trajectory['states'].append(final_state)

        total_reward += final_reward
        total_entropy += entropy_diff
        total_probs += prob_diff

        # Store trajectory summary
        trajectory['total_reward'] = total_reward
        trajectory['total_entropy'] = total_entropy
        trajectory['total_probs'] = total_probs
        trajectory['length'] = len(trajectory['actions'])
        trajectory['final_state'] = final_state

        trajectories.append(trajectory)

        if verbose:
            print(f"{traj_idx + 1:4d} | {total_reward:11.4f} | {total_entropy:12.4f} | "
                  f"{total_probs:10.4f} | {len(trajectory['actions']):6d} | {str(final_state)}")

    # Save trajectories if requested
    if save_trajectories:
        if output_file is None:
            output_file = f'generated_trajectories_{num_trajectories}.pkl'

        with open(output_file, 'wb') as f:
            pickle.dump(trajectories, f)

        if verbose:
            print(f"\nTrajectories saved to {output_file}")

    # Print summary statistics
    if verbose:
        rewards = [t['total_reward'] for t in trajectories]
        entropies = [t['total_entropy'] for t in trajectories]
        probs = [t['total_probs'] for t in trajectories]
        lengths = [t['length'] for t in trajectories]

        print(f"\nSummary Statistics:")
        print(f"Rewards - Mean: {np.mean(rewards):.4f}, Std: {np.std(rewards):.4f}")
        print(f"Entropies - Mean: {np.mean(entropies):.4f}, Std: {np.std(entropies):.4f}")
        print(f"Probs - Mean: {np.mean(probs):.4f}, Std: {np.std(probs):.4f}")
        print(f"Lengths - Mean: {np.mean(lengths):.2f}, Std: {np.std(lengths):.2f}")

        # Count final states
        final_states = [t['final_state'] for t in trajectories]
        state_counts = defaultdict(int)
        for state in final_states:
            state_counts[str(state)] += 1

        print(f"\nFinal State Distribution:")
        for state, count in state_counts.items():
            print(f"  {state}: {count}/{num_trajectories} ({count / num_trajectories * 100:.1f}%)")

    return trajectories


def analyze_trajectory(trajectory, trajectory_idx=0, detailed=False):
    """
    Analyze a single trajectory and print detailed information.

    Args:
        trajectory (dict): Single trajectory dictionary
        trajectory_idx (int): Index of trajectory for labeling
        detailed (bool): Whether to print step-by-step details
    """
    print(f"\n=== Trajectory {trajectory_idx} Analysis ===")
    print(f"Total Reward: {trajectory['total_reward']:.4f}")
    print(f"Total Entropy Change: {trajectory['total_entropy']:.4f}")
    print(f"Total Probability Change: {trajectory['total_probs']:.4f}")
    print(f"Length: {trajectory['length']} steps")
    print(f"Final State: {trajectory['final_state']}")

    if detailed:
        print(f"\nStep-by-step breakdown:")
        print("Step | State | Action | Observation | Reward")
        print("-" * 60)

        for i, (state, action, obs, reward) in enumerate(zip(
                trajectory['states'][:-1],
                trajectory['actions'],
                trajectory['observations'],
                trajectory['rewards']
        )):
            # Safely format state and observation strings
            state_str = str(state)[:15].ljust(15)
            action_str = str(action)[:6].ljust(6)
            obs_str = str(obs)[:15].ljust(15)

            print(f"{i + 1:4d} | {state_str} | {action_str} | {obs_str} | {reward:6.3f}")


# Example usage function
def main():
    """
    Example of how to use the trajectory generation function.
    """

    # Create environment
    env = prod_pomdp()

    # Generate trajectories using trained model
    model_path = 'ac_data/agent2_actor_critic_model_14.pth'  # Adjust path as needed

    trajectories = load_trained_agent_and_generate_trajectories(
        model_path=model_path,
        env=env,
        num_trajectories=20,
        max_steps=15,
        alpha=0,  # Same alpha used in training
        seed=42,  # For reproducibility
        save_trajectories=True,
        output_file='test_trajectories_2.pkl',
        verbose=True
    )

    # # Analyze first few trajectories in detail
    # for i in range(min(3, len(trajectories))):
    #     analyze_trajectory(trajectories[i], i, detailed=True)

    for i in range(len(trajectories)):
        print(trajectories[i]['states'])
        print(trajectories[i]['observations'])
        print(trajectories[i]['total_entropy'])

    return trajectories


if __name__ == "__main__":
    main()
