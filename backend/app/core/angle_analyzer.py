from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class JointStats:
    min_angle: float
    max_angle: float
    avg_angle: float
    std_dev: float
    range_of_motion: float

def analyze_angle_data(angle_data: List[Dict[str, float]], exercise_name: str) -> Dict[str, Any]:
    """
    Analyze angle data to extract key statistics for each joint.
    
    Args:
        angle_data (List[Dict[str, float]]): List of dictionaries containing joint angles for each frame
        exercise_name (str): Name of the exercise being performed
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - timestamp: When the analysis was performed
            - exercise_name: Name of the exercise
            - joint_stats: Dictionary of joint statistics
            - summary: A concise text summary of the analysis
    """
    if not angle_data:
        return {
            "timestamp": datetime.now().isoformat(),
            "exercise_name": exercise_name,
            "joint_stats": {},
            "summary": "No angle data available for analysis."
        }
    
    # Initialize dictionary to store all angles for each joint
    joint_angles = {}
    
    # Collect all angles for each joint
    for frame_angles in angle_data:
        for joint, angle in frame_angles.items():
            if joint not in joint_angles:
                joint_angles[joint] = []
            joint_angles[joint].append(angle)
    
    # Calculate statistics for each joint
    joint_stats = {}
    for joint, angles in joint_angles.items():
        angles_array = np.array(angles)
        stats = JointStats(
            min_angle=float(np.min(angles_array)),
            max_angle=float(np.max(angles_array)),
            avg_angle=float(np.mean(angles_array)),
            std_dev=float(np.std(angles_array)),
            range_of_motion=float(np.max(angles_array) - np.min(angles_array))
        )
        joint_stats[joint] = stats
    
    # Generate a concise summary
    summary_parts = [f"Exercise: {exercise_name}"]
    for joint, stats in joint_stats.items():
        summary_parts.append(
            f"{joint}: Range {stats.min_angle:.1f}° to {stats.max_angle:.1f}° "
            f"(Avg: {stats.avg_angle:.1f}°, ROM: {stats.range_of_motion:.1f}°)"
        )
    
    return {
        "timestamp": datetime.now().isoformat(),
        "exercise_name": exercise_name,
        "joint_stats": {
            joint: {
                "min_angle": stats.min_angle,
                "max_angle": stats.max_angle,
                "avg_angle": stats.avg_angle,
                "std_dev": stats.std_dev,
                "range_of_motion": stats.range_of_motion
            }
            for joint, stats in joint_stats.items()
        },
        "summary": "\n".join(summary_parts)
    } 