import pickle

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as patches


def create_grid_world_video(state_sequences, output_filename='grid_world_animation.gif',
                            fps=2, figsize=(10, 10), show_sensor_range=True,
                            output_format='gif'):
    """
    Create a video animation of UAV and UGV moving through a 6x6 grid world.

    Args:
        state_sequences (list): List of states, each state is ((ugv_x, ugv_y), (uav_x, uav_y))
        output_filename (str): Output video filename
        fps (int): Frames per second for the video
        figsize (tuple): Figure size for the plot
        show_sensor_range (bool): Whether to show UAV sensor range
        output_format (str): Output format ('gif', 'mp4', 'avi')
    """

    # Grid world parameters
    grid_size = 6
    obstacles = [(2, 1), (5, 1), (0, 2), (3, 3)]
    initial_ugv = (0, 3)
    initial_uav = (3, 0)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    def setup_grid():
        """Setup the basic grid layout"""
        ax.clear()
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')

        # Draw grid lines
        for i in range(grid_size + 1):
            ax.axhline(y=i - 0.5, color='lightgray', linewidth=0.5)
            ax.axvline(x=i - 0.5, color='lightgray', linewidth=0.5)

        # Draw obstacles
        for obs_x, obs_y in obstacles:
            obstacle = Rectangle((obs_x - 0.4, obs_y - 0.4), 0.8, 0.8,
                                 facecolor='black', edgecolor='black')
            ax.add_patch(obstacle)

        # Add grid coordinates as text
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) not in obstacles:
                    ax.text(i, j, f'({i},{j})', ha='center', va='center',
                            fontsize=8, alpha=0.3)

        # Labels and title
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title('UAV and UGV Navigation in 6x6 Grid World', fontsize=14, fontweight='bold')

        # Invert y-axis to match typical grid coordinate system
        ax.invert_yaxis()

        return ax

    def draw_sensor_range(uav_pos, alpha=0.2):
        """Draw UAV sensor range (3x3 area around UAV)"""
        uav_x, uav_y = uav_pos
        sensor_positions = []

        # Get all positions in 3x3 area around UAV (including diagonals)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                sensor_x = uav_x + dx
                sensor_y = uav_y + dy
                if 0 <= sensor_x < grid_size and 0 <= sensor_y < grid_size:
                    sensor_positions.append((sensor_x, sensor_y))

        # Draw sensor range
        for pos_x, pos_y in sensor_positions:
            sensor_cell = Rectangle((pos_x - 0.5, pos_y - 0.5), 1, 1,
                                    facecolor='yellow', alpha=alpha, edgecolor='orange')
            ax.add_patch(sensor_cell)

        return sensor_positions

    def animate_frame(frame_num):
        """Animation function for each frame"""
        setup_grid()

        if frame_num < len(state_sequences):
            ugv_pos, uav_pos = state_sequences[frame_num]

            # Draw sensor range if enabled
            if show_sensor_range:
                sensor_positions = draw_sensor_range(uav_pos)

            # Draw UGV (Ground Vehicle) - Blue square
            ugv_x, ugv_y = ugv_pos
            ugv_marker = Rectangle((ugv_x - 0.3, ugv_y - 0.3), 0.6, 0.6,
                                   facecolor='blue', edgecolor='darkblue', linewidth=2)
            ax.add_patch(ugv_marker)
            ax.text(ugv_x, ugv_y - 0.6, 'UGV', ha='center', va='top',
                    fontweight='bold', color='blue', fontsize=10)

            # Draw UAV (Air Vehicle) - Red circle
            uav_x, uav_y = uav_pos
            uav_marker = Circle((uav_x, uav_y), 0.3, facecolor='red',
                                edgecolor='darkred', linewidth=2)
            ax.add_patch(uav_marker)
            ax.text(uav_x, uav_y + 0.6, 'UAV', ha='center', va='bottom',
                    fontweight='bold', color='red', fontsize=10)

            # Add frame information
            ax.text(0.02, 0.98, f'Step: {frame_num + 1}/{len(state_sequences)}',
                    transform=ax.transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Add position information
            info_text = f'UGV: {ugv_pos}\nUAV: {uav_pos}'
            ax.text(0.02, 0.85, info_text, transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            # Draw trajectory lines if more than one frame
            if frame_num > 0:
                # UGV trajectory
                ugv_trajectory_x = [state[0][0] for state in state_sequences[:frame_num + 1]]
                ugv_trajectory_y = [state[0][1] for state in state_sequences[:frame_num + 1]]
                ax.plot(ugv_trajectory_x, ugv_trajectory_y, 'b--', alpha=0.5, linewidth=2, label='UGV Path')

                # UAV trajectory
                uav_trajectory_x = [state[1][0] for state in state_sequences[:frame_num + 1]]
                uav_trajectory_y = [state[1][1] for state in state_sequences[:frame_num + 1]]
                ax.plot(uav_trajectory_x, uav_trajectory_y, 'r--', alpha=0.5, linewidth=2, label='UAV Path')

        # Add legend
        legend_elements = [
            patches.Patch(color='blue', label='UGV (Ground Vehicle)'),
            patches.Patch(color='red', label='UAV (Air Vehicle)'),
            patches.Patch(color='black', label='Obstacles'),
        ]
        if show_sensor_range:
            legend_elements.append(patches.Patch(color='yellow', alpha=0.5, label='UAV Sensor Range'))

        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))

        return ax.patches + ax.texts + ax.lines

    # Create animation
    num_frames = len(state_sequences)
    ani = animation.FuncAnimation(fig, animate_frame, frames=num_frames,
                                  interval=1000 // fps, blit=False, repeat=True)

    # Save animation with better error handling
    print(f"Creating {output_format.upper()} with {num_frames} frames...")

    try:
        if output_format.lower() == 'gif':
            # Use PillowWriter for GIF (most reliable)
            ani.save(output_filename, writer='pillow', fps=fps)
            print(f"GIF saved as: {output_filename}")

        elif output_format.lower() == 'mp4':
            # Try different MP4 writers in order of preference
            writers_to_try = [
                ('ffmpeg', {'codec': 'libx264', 'extra_args': ['-pix_fmt', 'yuv420p']}),
                ('ffmpeg', {'codec': 'mpeg4'}),
                ('html', {})  # Fallback to HTML animation
            ]

            success = False
            for writer_name, writer_args in writers_to_try:
                try:
                    if writer_name == 'html':
                        html_filename = output_filename.replace('.mp4', '.html')
                        ani.save(html_filename, writer='html')
                        print(f"HTML animation saved as: {html_filename}")
                        success = True
                        break
                    else:
                        Writer = animation.writers[writer_name]
                        writer = Writer(fps=fps, metadata=dict(artist='Grid World Visualizer'),
                                        bitrate=1800, **writer_args)
                        ani.save(output_filename, writer=writer)
                        print(f"MP4 saved as: {output_filename}")
                        success = True
                        break
                except Exception as e:
                    print(f"Failed with {writer_name}: {str(e)}")
                    continue

            if not success:
                print("All MP4 writers failed. Falling back to GIF...")
                gif_filename = output_filename.replace('.mp4', '.gif')
                ani.save(gif_filename, writer='pillow', fps=fps)
                print(f"GIF saved as: {gif_filename}")

        else:
            # Default to GIF for other formats
            ani.save(output_filename, writer='pillow', fps=fps)
            print(f"Animation saved as: {output_filename}")

    except Exception as e:
        print(f"Error saving animation: {str(e)}")
        print("Trying alternative GIF format...")
        gif_filename = output_filename.replace('.mp4', '.gif').replace('.avi', '.gif')
        try:
            ani.save(gif_filename, writer='pillow', fps=fps)
            print(f"GIF saved as: {gif_filename}")
        except Exception as gif_error:
            print(f"GIF also failed: {str(gif_error)}")
            print("Showing animation instead of saving...")

    # Show the animation (optional)
    plt.tight_layout()
    plt.show()

    return ani


def create_static_visualization(state_sequences, save_path=None, show_all_positions=True):
    """
    Create a static visualization showing all positions visited.

    Args:
        state_sequences (list): List of states
        save_path (str): Path to save the static image
        show_all_positions (bool): Whether to show all visited positions
    """
    grid_size = 6
    obstacles = [(2, 1), (5, 1), (0, 2), (3, 3)]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Setup grid
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')

    # Draw grid lines
    for i in range(grid_size + 1):
        ax.axhline(y=i - 0.5, color='lightgray', linewidth=0.5)
        ax.axvline(x=i - 0.5, color='lightgray', linewidth=0.5)

    # Draw obstacles
    for obs_x, obs_y in obstacles:
        obstacle = Rectangle((obs_x - 0.4, obs_y - 0.4), 0.8, 0.8,
                             facecolor='black', edgecolor='black')
        ax.add_patch(obstacle)

    if show_all_positions and len(state_sequences) > 1:
        # Draw full trajectories
        ugv_trajectory_x = [state[0][0] for state in state_sequences]
        ugv_trajectory_y = [state[0][1] for state in state_sequences]
        ax.plot(ugv_trajectory_x, ugv_trajectory_y, 'b-', linewidth=3, alpha=0.7, label='UGV Path')

        uav_trajectory_x = [state[1][0] for state in state_sequences]
        uav_trajectory_y = [state[1][1] for state in state_sequences]
        ax.plot(uav_trajectory_x, uav_trajectory_y, 'r-', linewidth=3, alpha=0.7, label='UAV Path')

        # Mark all visited positions
        for i, (ugv_pos, uav_pos) in enumerate(state_sequences):
            # UGV positions
            ax.plot(ugv_pos[0], ugv_pos[1], 'bo', markersize=8, alpha=0.6)
            ax.text(ugv_pos[0] + 0.1, ugv_pos[1] + 0.1, str(i), fontsize=8, color='blue')

            # UAV positions
            ax.plot(uav_pos[0], uav_pos[1], 'ro', markersize=8, alpha=0.6)
            ax.text(uav_pos[0] + 0.1, uav_pos[1] - 0.1, str(i), fontsize=8, color='red')

    # Mark initial and final positions
    if state_sequences:
        initial_ugv, initial_uav = state_sequences[0]
        final_ugv, final_uav = state_sequences[-1]

        # Initial positions (larger markers)
        ax.plot(initial_ugv[0], initial_ugv[1], 'bs', markersize=15, label='UGV Start')
        ax.plot(initial_uav[0], initial_uav[1], 'rs', markersize=15, label='UAV Start')

        # Final positions (star markers)
        ax.plot(final_ugv[0], final_ugv[1], 'b*', markersize=20, label='UGV End')
        ax.plot(final_uav[0], final_uav[1], 'r*', markersize=20, label='UAV End')

    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('UAV and UGV Complete Trajectory', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Static visualization saved as: {save_path}")

    plt.show()
    return fig, ax


# Example usage and test function
def example_usage():
    """Example of how to use the video generation functions"""

    # Example state sequence (replace with your actual data)
    example_states = [
        ((0, 3), (3, 0)),  # Initial positions
        ((0, 2), (3, 1)),  # Step 1
        ((1, 2), (2, 1)),  # Step 2 - Note: UAV can't be on obstacle, so adjusted
        ((1, 3), (2, 2)),  # Step 3
        ((2, 3), (1, 2)),  # Step 4
        ((2, 4), (1, 3)),  # Step 5
        ((3, 4), (2, 3)),  # Step 6
        ((4, 4), (3, 4)),  # Step 7
        ((4, 3), (4, 4)),  # Step 8
        ((5, 3), (5, 4)),  # Final positions
    ]

    print("Creating GIF animation (most reliable)...")
    ani = create_grid_world_video(
        state_sequences=example_states,
        output_filename='grid_world_demo.gif',
        fps=1,  # Slow animation for demo
        show_sensor_range=True,
        output_format='gif'
    )

    print("Creating static visualization...")
    create_static_visualization(
        state_sequences=example_states,
        save_path='grid_world_static.png',
        show_all_positions=True
    )

    return example_states


# Function to use with your trajectory data
def visualize_trajectory_data(trajectories, trajectory_index=0, output_dir='./videos/',
                              output_format='gif'):
    """
    Visualize trajectory data from your trained agent.

    Args:
        trajectories (list): List of trajectory dictionaries from your agent
        trajectory_index (int): Which trajectory to visualize
        output_dir (str): Directory to save output files
        output_format (str): Output format ('gif', 'mp4', 'avi')
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if trajectory_index >= len(trajectories):
        print(f"Trajectory index {trajectory_index} out of range. Available: 0-{len(trajectories) - 1}")
        return

    trajectory = trajectories[trajectory_index]
    state_sequences = trajectory['states']

    # Convert your state format to the expected format if needed
    # Assuming your states are in the format: ((ugv_pos), (uav_pos), type, auto_st)
    # Extract just the positions
    formatted_states = []
    for state in state_sequences:
        if isinstance(state, tuple) and len(state) >= 2:
            # Extract UGV and UAV positions
            if isinstance(state[0], tuple) and len(state[0]) >= 2:
                ugv_pos = state[0][0]  # UGV position
                uav_pos = state[0][1]  # UAV position
                formatted_states.append((ugv_pos, uav_pos))

    if not formatted_states:
        print("Could not extract position data from trajectory states")
        return

    print(f"Visualizing trajectory {trajectory_index} with {len(formatted_states)} states...")

    # Create video/animation
    if output_format == 'gif':
        video_filename = os.path.join(output_dir, f'trajectory_{trajectory_index}_animation.gif')
    else:
        video_filename = os.path.join(output_dir, f'trajectory_{trajectory_index}_video.{output_format}')

    ani = create_grid_world_video(
        state_sequences=formatted_states,
        output_filename=video_filename,
        fps=2,
        show_sensor_range=True,
        output_format=output_format
    )

    # Create static visualization
    static_filename = os.path.join(output_dir, f'trajectory_{trajectory_index}_static.png')
    create_static_visualization(
        state_sequences=formatted_states,
        save_path=static_filename,
        show_all_positions=True
    )

    print(f"Visualization complete for trajectory {trajectory_index}")
    return formatted_states


def read_states(trajectories_file_path, trajectory_number):
    states = []
    with open(trajectories_file_path, 'rb') as file:
        trajectories = pickle.load(file)
    true_states = trajectories[trajectory_number]['states']
    observations = trajectories[trajectory_number]['observations']
    total_entropy = trajectories[trajectory_number]['total_entropy']
    for true_state in true_states:
        if true_state != 'sink1' and true_state != 'sink2' and true_state != 'sink3':
            state = (true_state[0][0], true_state[0][1])
        else:
            state = states[-1]
        states.append(state)
    return states, true_states, observations, total_entropy


def main():
    states, true_states, observations, total_entropy = read_states('test_trajectories.pkl', 14)
    print("The information about this trajectory:")
    print("The states are:")
    print(true_states)
    print("The observations are:")
    print(observations)
    print("The total entropy is:")
    print(total_entropy)

    print("Creating GIF animation (most reliable)...")
    ani = create_grid_world_video(
        state_sequences=states,
        output_filename='grid_world_demo.gif',
        fps=1,  # Slow animation for demo
        show_sensor_range=True,
        output_format='gif'
    )

    print("Creating static visualization...")
    create_static_visualization(
        state_sequences=states,
        save_path='grid_world_static.png',
        show_all_positions=True
    )

    return states


if __name__ == "__main__":
    # Run example
    main()
