from fastapi import APIRouter, UploadFile, File, HTTPException, Response, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from ..core.pose_analyzer import PoseAnalyzer
from ..core.angle_analyzer import analyze_angle_data
import tempfile
import os
import json
import base64
from typing import Dict, Any

class ExerciseDataStore:
    def __init__(self):
        self._data: Dict[str, Any] = {}
    
    def store_data(self, data: Dict[str, Any]):
        self._data = data
    
    def get_data(self) -> Dict[str, Any]:
        return self._data
    
    def has_data(self) -> bool:
        return bool(self._data)

# Create a singleton instance
exercise_store = ExerciseDataStore()

router = APIRouter()
pose_analyzer = PoseAnalyzer()

# Ensure uploads directory exists
UPLOADS_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Store the last processed video's data
last_processed_data = {"video_path": None, "exercise_data": None}

def _cleanup_files(tmp_video_path: Path, output_path: Path | None):
    """Helper function to clean up temporary files in a background task."""
    print(f"[Backend] Attempting to clean up files: {tmp_video_path} and {output_path}")
    if tmp_video_path.exists():
        try:
            os.unlink(tmp_video_path)
            print(f"[Backend] Successfully deleted temporary input file: {tmp_video_path}")
        except Exception as e:
            print(f"[Backend] Error deleting temporary input file {tmp_video_path}: {e}")
    
    if output_path and output_path.exists():
        try:
            os.unlink(output_path)
            print(f"[Backend] Successfully deleted temporary output file: {output_path}")
        except Exception as e:
            print(f"[Backend] Error deleting temporary output file {output_path}: {e}")

@router.post("/process")
async def process_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    print(f"[Backend] Received video file: {file.filename}")
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only MP4, AVI, MOV are allowed.")

    # Initialize output_path to None to prevent UnboundLocalError
    output_path = None

    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video_file:
        tmp_video_path = Path(tmp_video_file.name)
        contents = await file.read()
        tmp_video_file.write(contents)
    print(f"[Backend] Video saved temporarily to: {tmp_video_path}")

    try:
        output_path, angle_data, avg_angles, classified_exercise, feedback = pose_analyzer.process_video(str(tmp_video_path))
        print(f"[Backend] Video processed. Output path: {output_path}")
        print(f"[Backend] Generated feedback: {feedback}")

        # Analyze angle data
        angle_analysis = analyze_angle_data(angle_data, classified_exercise)
        print(f"[Backend] Generated angle analysis: {angle_analysis['summary']}")

        # Read the processed video file and encode it to base64
        with open(output_path, "rb") as video_file:
            encoded_video = base64.b64encode(video_file.read()).decode("utf-8")
        print(f"[Backend] Encoded video to base64. Length: {len(encoded_video)} characters")

        exercise_data = {
            "angle_data": angle_data,
            "average_angles": avg_angles,
            "classified_exercise": classified_exercise,
            "feedback": feedback,
            "angle_analysis": angle_analysis
        }
        
        print(f"[Backend] Generated exercise data: {json.dumps(exercise_data, indent=2)}")

        # Update the global store
        last_processed_data["video_path"] = output_path
        last_processed_data["exercise_data"] = exercise_data
        
        response_data = {
            "video_data": encoded_video,
            "exercise_data": exercise_data
        }

        print(f"[Backend] Sending JSON response with video_data length: {len(response_data['video_data'])} and exercise_data: {response_data['exercise_data']['classified_exercise']}")
        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"[Backend] Error during video processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {e}")

    finally:
        # Add cleanup task to background tasks
        # For debugging, we are temporarily NOT cleaning up output_path
        background_tasks.add_task(_cleanup_files, tmp_video_path, None) # Only clean up input temp file for now
        print(f"[Backend] Processed video saved temporarily for inspection at: {output_path}")

@router.get("/exercise-data")
async def get_exercise_data():
    if not last_processed_data["exercise_data"]:
        print("[Backend] No exercise data available. Returning 404.")
        raise HTTPException(status_code=404, detail="No exercise data available. Please process a video first.")
    print("[Backend] Returning last processed exercise data.")
    return JSONResponse(content=last_processed_data["exercise_data"]) 