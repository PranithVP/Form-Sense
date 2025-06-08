from typing import Dict, List, Optional
import json

def extract_exercise_data(header_data: str) -> Dict:
    """
    Extract and format exercise data from the X-Exercise-Data header.
    
    Args:
        header_data (str): The raw X-Exercise-Data header content
        
    Returns:
        Dict: A formatted dictionary containing:
            - exercise_name: The classified exercise
            - average_angles: Dict of joint angles
            - feedback: List of feedback points
            - angle_data: List of frame-by-frame angles (optional)
    """
    try:
        # Parse the JSON data from the header
        data = json.loads(header_data)
        
        # Extract the key information
        formatted_data = {
            "exercise_name": data.get("classified_exercise", "unknown"),
            "average_angles": data.get("average_angles", {}),
            "feedback": data.get("feedback", []),
            "angle_data": data.get("angle_data", [])  # Optional: include if needed for detailed analysis
        }
        
        return formatted_data
        
    except json.JSONDecodeError:
        return {
            "exercise_name": "unknown",
            "average_angles": {},
            "feedback": ["Error: Could not parse exercise data"],
            "angle_data": []
        }

def get_formatted_feedback(data: Dict) -> str:
    """
    Format the exercise data into a concise string for LLM processing.
    
    Args:
        data (Dict): The formatted exercise data from extract_exercise_data
        
    Returns:
        str: A formatted string containing the key information
    """
    exercise_name = data["exercise_name"]
    avg_angles = data["average_angles"]
    feedback = data["feedback"]
    
    # Format the output
    output = f"Exercise: {exercise_name}\n\n"
    
    # Add average angles
    output += "Average Joint Angles:\n"
    for joint, angle in avg_angles.items():
        output += f"- {joint}: {angle:.1f}°\n"
    
    # Add feedback
    output += "\nForm Feedback:\n"
    for point in feedback:
        output += f"- {point}\n"
    
    return output 