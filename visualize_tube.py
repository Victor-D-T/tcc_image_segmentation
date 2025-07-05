import json
import numpy as np

def analyze_tube_structure(json_file):
    """Analyze the tube structure and sessions"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    session_positions = data['session_positions']
    drone_path = np.array(data['drone_path'])
    
    if len(session_positions) < 2:
        print("❌ Need at least 2 sessions for analysis")
        return
    
    # Convert to numpy array for easier calculation
    sessions = np.array(list(session_positions.values()))
    session_ids = list(session_positions.keys())
    
    # Calculate distances between consecutive sessions
    session_distances = []
    for i in range(len(sessions) - 1):
        dist = np.linalg.norm(sessions[i+1] - sessions[i])
        session_distances.append(dist)
        print(f"Distance between Session {session_ids[i]} and {session_ids[i+1]}: {dist:.2f}m")
    
    # Calculate tube statistics
    avg_session_distance = np.mean(session_distances)
    tube_length = np.sum(session_distances)
    
    print(f"\n📏 Tube Analysis:")
    print(f"   Total tube length: {tube_length:.2f} meters")
    print(f"   Average session spacing: {avg_session_distance:.2f} meters")
    print(f"   Number of segments: {len(session_distances)}")
    
    # Detect potential issues
    std_distance = np.std(session_distances)
    if std_distance > avg_session_distance * 0.3:
        print(f"⚠️  Warning: Irregular session spacing detected (std: {std_distance:.2f})")
    
    return {
        'tube_length': tube_length,
        'avg_session_distance': avg_session_distance,
        'session_distances': session_distances,
        'num_sessions': len(session_positions)
    }

# Run the analysis
analysis = analyze_tube_structure('tube_3d.json')