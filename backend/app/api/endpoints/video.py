from fastapi import UploadFile, File, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import APIRouter, Depends, BackgroundTasks
import tempfile
import os
import time
from pathlib import Path
from backend.utils.auth import get_current_user
from backend.utils.pose_analyzer import PoseAnalyzer
from backend.models.user import User

router = APIRouter()
pose_analyzer = PoseAnalyzer()

# Ensure uploads directory exists
UPLOADS_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

def _cleanup_processed_file(file_path: Path):
    """Helper function to clean up processed files after a delay."""
    time.sleep(5)  # Wait 5 seconds to ensure file is no longer needed
    if file_path.exists():
        try:
            os.unlink(file_path)
            print(f"[Video Endpoint] Successfully deleted processed file: {file_path}")
        except Exception as e:
            print(f"[Video Endpoint] Error deleting processed file {file_path}: {e}")

@router.post("/process")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Process a video file and return the path to the processed video, angle data, classified exercise, and feedback."""
    try:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Process the video
        output_path, angle_data, classified_exercise, feedback = pose_analyzer.process_video(temp_path)
        
        # Move the processed video to the uploads directory
        output_filename = Path(output_path).name
        final_output_path = UPLOADS_DIR / output_filename
        os.rename(output_path, final_output_path)
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        # Add cleanup task for the processed file after a delay
        background_tasks.add_task(_cleanup_processed_file, final_output_path)
        
        # Calculate average angles from angle_data
        avg_angles = {}
        if angle_data:
            for frame_angles in angle_data:
                for joint, angle in frame_angles.items():
                    if joint not in avg_angles:
                        avg_angles[joint] = []
                    avg_angles[joint].append(angle)
            
            # Calculate averages
            for joint in avg_angles:
                avg_angles[joint] = sum(avg_angles[joint]) / len(avg_angles[joint])
        
        return {
            "message": "Video processed successfully",
            "output_path": str(final_output_path),
            "angle_data": angle_data,
            "average_angles": avg_angles,
            "classified_exercise": classified_exercise,
            "feedback": feedback
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 