import json
import numpy as np


def export_to_formats(json_file, output_prefix='tube_reconstruction'):
    """Export reconstruction to various formats"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    session_positions = data['session_positions']
    drone_path = np.array(data['drone_path'])
    
    # Export to CSV for spreadsheet analysis
    import pandas as pd
    
    # Sessions CSV
    if session_positions:
        sessions_df = pd.DataFrame([
            {'session_id': sid, 'x': pos[0], 'y': pos[1], 'z': pos[2]}
            for sid, pos in session_positions.items()
        ])
        sessions_df.to_csv(f'{output_prefix}_sessions.csv', index=False)
        print(f"📄 Sessions exported to: {output_prefix}_sessions.csv")
    
    # Drone path CSV
    if len(drone_path) > 0:
        path_df = pd.DataFrame(drone_path, columns=['x', 'y', 'z'])
        path_df['frame'] = range(len(drone_path))
        path_df.to_csv(f'{output_prefix}_drone_path.csv', index=False)
        print(f"📄 Drone path exported to: {output_prefix}_drone_path.csv")
    
    # Export to PLY format (for 3D software like Blender, MeshLab)
    def export_to_ply(filename, points, colors=None):
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if colors:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for i, point in enumerate(points):
                if colors:
                    f.write(f"{point[0]} {point[1]} {point[2]} {colors[i][0]} {colors[i][1]} {colors[i][2]}\n")
                else:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")
    
    # Combine sessions and path for PLY export
    all_points = []
    all_colors = []
    
    # Add sessions (red)
    for pos in session_positions.values():
        all_points.append(pos)
        all_colors.append([255, 0, 0])  # Red
    
    # Add drone path (blue)
    for pos in drone_path:
        all_points.append(pos)
        all_colors.append([0, 0, 255])  # Blue
    
    if all_points:
        export_to_ply(f'{output_prefix}.ply', all_points, all_colors)
        print(f"📄 3D model exported to: {output_prefix}.ply")

# Export to different formats
export_to_formats('tube_3d.json')

def create_interactive_dashboard(json_file):
    """Create an interactive dashboard using plotly"""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:
        print("❌ Please install plotly: pip install plotly")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    session_positions = data['session_positions']
    drone_path = np.array(data['drone_path'])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scatter3d", "colspan": 2}, None],
               [{"type": "scatter"}, {"type": "scatter"}]],
        subplot_titles=('3D Tube Reconstruction', 'Drone Path (X-Y)', 'Drone Path (Z over time)')
    )
    
    # 3D plot
    if session_positions:
        session_coords = np.array(list(session_positions.values()))
        fig.add_trace(
            go.Scatter3d(
                x=session_coords[:, 0], y=session_coords[:, 1], z=session_coords[:, 2],
                mode='markers+text',
                marker=dict(size=10, color='red'),
                text=[f'S{sid}' for sid in session_positions.keys()],
                name='Sessions'
            ), row=1, col=1
        )
    
    if len(drone_path) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=drone_path[:, 0], y=drone_path[:, 1], z=drone_path[:, 2],
                mode='lines+markers',
                line=dict(color='blue', width=4),
                marker=dict(size=3),
                name='Drone Path'
            ), row=1, col=1
        )
        
        # 2D projections - FIX: Convert range to list
        fig.add_trace(
            go.Scatter(x=drone_path[:, 0], y=drone_path[:, 1], 
                      mode='lines', name='X-Y Path'), row=2, col=1
        )
        
        # FIX: Convert range to list for frame indices
        frame_indices = list(range(len(drone_path)))
        fig.add_trace(
            go.Scatter(x=frame_indices, y=drone_path[:, 2], 
                      mode='lines', name='Z over time'), row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="Tube Reconstruction Dashboard")
    fig.show()

create_interactive_dashboard('tube_3d.json')