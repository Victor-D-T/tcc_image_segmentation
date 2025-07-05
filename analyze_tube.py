import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_tube_reconstruction(json_file):
    """Visualize the 3D tube reconstruction"""
    
    # Load the data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    session_positions = data['session_positions']
    drone_path = np.array(data['drone_path'])
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot session dividers
    if session_positions:
        session_coords = np.array(list(session_positions.values()))
        ax.scatter(session_coords[:, 0], session_coords[:, 1], session_coords[:, 2], 
                  c='red', s=100, alpha=0.8, label='Session Dividers')
        
        # Add session IDs as labels
        for session_id, pos in session_positions.items():
            ax.text(pos[0], pos[1], pos[2], f'S{session_id}', fontsize=8)
    
    # Plot drone path
    if len(drone_path) > 0:
        ax.plot(drone_path[:, 0], drone_path[:, 1], drone_path[:, 2], 
               'b-', linewidth=2, alpha=0.7, label='Drone Path')
        
        # Mark start and end
        ax.scatter(drone_path[0, 0], drone_path[0, 1], drone_path[0, 2], 
                  c='green', s=200, marker='^', label='Start')
        ax.scatter(drone_path[-1, 0], drone_path[-1, 1], drone_path[-1, 2], 
                  c='orange', s=200, marker='v', label='End')
    
    # Set labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('3D Tube Reconstruction')
    ax.legend()
    
    # Make it look better
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"📊 Reconstruction Statistics:")
    print(f"   Sessions detected: {len(session_positions)}")
    print(f"   Drone path length: {len(drone_path)} points")
    if len(drone_path) > 1:
        total_distance = np.sum(np.linalg.norm(np.diff(drone_path, axis=0), axis=1))
        print(f"   Total distance traveled: {total_distance:.2f} meters")

# Run the visualization
visualize_tube_reconstruction('tube_3d.json')