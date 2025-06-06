import torch
import numpy as np
import random


def evaluate_model_entropy(model_path, env, info_rewards, num_episodes=100, max_steps=20):
    """
    Simple function to evaluate average entropy from a saved model

    Args:
        model_path: Path to the saved .pth model file
        env: Environment instance (prod_pomdp)
        info_rewards: InformationRewards instance
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode

    Returns:
        float: Average total entropy over all episodes
    """

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    # Create agent (adjust parameters to match your training if needed)
    from actor_critic import Agent2ActorCritic  # Update this import
    agent = Agent2ActorCritic(env, T=10, hidden_size=128)

    # Load weights
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.actor.eval()

    total_entropies = []

    with torch.no_grad():
        for episode in range(num_episodes):
            episode_obs = []
            episode_as = []

            # Initialize episode
            state = random.choices(env.initial_states, env.initial_dist_sampling, k=1)[0]

            # Run episode
            for step in range(max_steps):
                # Get action
                obs_sequence = agent.prepare_observation_sequence(episode_obs)
                action_probs = agent.actor(obs_sequence)
                agent2_action = torch.argmax(action_probs).item()
                act = env.actions[agent2_action]
                episode_as.append(agent2_action)

                # Get observation and next state
                obs = env.observation_function_sampler(state, act)
                episode_obs.append(obs)
                state = env.next_state_sampler(state, act)

                # Check if done
                if state in env.sink_states:
                    break

            # Final step
            act = 'e'
            agent2_action = env.actions.index(act)
            episode_as.append(agent2_action)
            final_state = env.next_state_sampler(state, act)
            obs_final = env.observation_function_sampler(final_state, act)
            episode_obs.append(obs_final)

            # Get entropy for this episode
            entropy_diff, _ = info_rewards.reward_function(episode_obs, episode_as)
            total_entropies.append(entropy_diff)

    # Return average entropy
    return -np.mean(total_entropies)


# Simple usage
def main():
    from product_pomdp import prod_pomdp
    from information_rewards import InformationRewards

    env = prod_pomdp()
    info_rewards = InformationRewards()

    avg_entropy = evaluate_model_entropy(
        model_path='agent2_actor_critic_model.pth',
        env=env,
        info_rewards=info_rewards,
        num_episodes=1000
    )

    print(f"Average total entropy over 1000 episodes: {avg_entropy:.4f}")
    return avg_entropy


if __name__ == "__main__":
    main()