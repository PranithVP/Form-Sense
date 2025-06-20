import cv2
import mediapipe as mp
import numpy as np
import math
from pathlib import Path
import os
from collections import Counter
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
import timm

class LandmarkSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []

    def smooth(self, landmarks):
        # landmarks: list of landmark objects (with x, y, z, visibility)
        arr = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
        self.history.append(arr)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        smoothed = np.mean(self.history, axis=0)
        # Return as list of objects with x, y, z, visibility attributes
        class LM:
            pass
        smoothed_landmarks = []
        for i in range(smoothed.shape[0]):
            lm = LM()
            lm.x, lm.y, lm.z, lm.visibility = smoothed[i]
            smoothed_landmarks.append(lm)
        return smoothed_landmarks

class PoseAnalyzer:
    def __init__(self):
        # Define exercise classes first
        self.exercise_classes = [
            "Barbell Biceps Curl",
            "Bench Press",
            "Chest Fly Machine",
            "Deadlift",
            "Decline Bench Press",
            "Hammer Curl",
            "Hip Thrust",
            "Incline Bench Press",
            "Lat Pulldown",
            "Lateral Raises",
            "Leg Extension",
            "Leg Raises",
            "Plank",
            "Pull Up",
            "Push Up",
            "Romanian Deadlift",
            "Russian Twist",
            "Shoulder Press",
            "Squat",
            "T Bar Row",
            "Tricep Dips",
            "Tricep Pushdown"
        ]
        
        # Define ideal angles for each exercise
        self.ideal_angles = {
            "Barbell Biceps Curl": {
                "Left Elbow": 60,    # Deep flexion
                "Right Elbow": 60,
                "Left Shoulder": 20, # Slight forward tilt
                "Right Shoulder": 20,
                "Left Hip": 180,     # Standing tall
                "Right Hip": 180,
                "Left Knee": 180,
                "Right Knee": 180
            },
            "Bench Press": {
                "Left Elbow": 90,    # Arms at 90 degrees
                "Right Elbow": 90,
                "Left Shoulder": 90, # Shoulders at 90 degrees
                "Right Shoulder": 90,
                "Left Hip": 180,     # Lying flat
                "Right Hip": 180,
                "Left Knee": 90,     # Feet flat on ground
                "Right Knee": 90
            },
            "Deadlift": {
                "Left Elbow": 180,
                "Right Elbow": 180,
                "Left Shoulder": 180,
                "Right Shoulder": 180,
                "Left Hip": 180,
                "Right Hip": 180,
                "Left Knee": 180,
                "Right Knee": 180
            },
            "Plank": {
                "Left Elbow": 180,   # Arms extended
                "Right Elbow": 180,
                "Left Shoulder": 30, # Slight angle from torso
                "Right Shoulder": 30,
                "Left Hip": 180,     # Body in a straight line
                "Right Hip": 180,
                "Left Knee": 180,
                "Right Knee": 180
            },
            "Squat": {
                "Left Elbow": 180,
                "Right Elbow": 180,
                "Left Shoulder": 180,
                "Right Shoulder": 180,
                "Left Hip": 90,      # Deep squat position
                "Right Hip": 90,
                "Left Knee": 90,
                "Right Knee": 90
            }
        }
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model
        self.model = self.get_model()
        self.model_loaded = False  # Flag to check if the model is loaded
        model_path = Path(__file__).parent.parent / "models" / "best_model.pth"
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.model_loaded = True
            except Exception as e:
                print(f"Warning: Error loading model file at {model_path}: {e}")
        else:
            print(f"Warning: Model file not found at {model_path}. Exercise classification will be disabled.")
        
        # Initialize MediaPipe Pose with improved configuration for video
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # Important for video!
            model_complexity=1,
            smooth_landmarks=True,    # Enable smoothing
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7
        )
        
        # Define angles to calculate
        self.ANGLES_TO_CALCULATE = {
            "Left Elbow": (self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                          self.mp_pose.PoseLandmark.LEFT_ELBOW,
                          self.mp_pose.PoseLandmark.LEFT_WRIST),
            "Right Elbow": (self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                           self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                           self.mp_pose.PoseLandmark.RIGHT_WRIST),
            "Left Shoulder": (self.mp_pose.PoseLandmark.LEFT_ELBOW,
                             self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                             self.mp_pose.PoseLandmark.LEFT_HIP),
            "Right Shoulder": (self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                              self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                              self.mp_pose.PoseLandmark.RIGHT_HIP),
            "Left Hip": (self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                        self.mp_pose.PoseLandmark.LEFT_HIP,
                        self.mp_pose.PoseLandmark.LEFT_KNEE),
            "Right Hip": (self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                         self.mp_pose.PoseLandmark.RIGHT_HIP,
                         self.mp_pose.PoseLandmark.RIGHT_KNEE),
            "Left Knee": (self.mp_pose.PoseLandmark.LEFT_HIP,
                         self.mp_pose.PoseLandmark.LEFT_KNEE,
                         self.mp_pose.PoseLandmark.LEFT_ANKLE),
            "Right Knee": (self.mp_pose.PoseLandmark.RIGHT_HIP,
                          self.mp_pose.PoseLandmark.RIGHT_KNEE,
                          self.mp_pose.PoseLandmark.RIGHT_ANKLE),
        }

    def get_model(self):
        # Use EfficientNet (must match the architecture used in training)
        model = timm.create_model('efficientnet_b3a', pretrained=False, num_classes=len(self.exercise_classes))
        return model.to(self.device)

    def preprocess_frame(self, frame):
        """Preprocess a frame for the neural network."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        transform_pipeline = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        return transform_pipeline(image)

    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def get_joint_angles(self, landmarks, image_shape, visibility_threshold=0.6):
        """Calculate angles for all defined joints."""
        angles = {}
        for angle_name, (a, b, c) in self.ANGLES_TO_CALCULATE.items():
            # Get landmark points
            point_a = landmarks[a.value]
            point_b = landmarks[b.value]
            point_c = landmarks[c.value]
            
            # Check visibility
            if (point_a.visibility > visibility_threshold and 
                point_b.visibility > visibility_threshold and 
                point_c.visibility > visibility_threshold):
                
                # Convert normalized coordinates to pixel coordinates
                h, w = image_shape[:2]
                a_coords = (int(point_a.x * w), int(point_a.y * h))
                b_coords = (int(point_b.x * w), int(point_b.y * h))
                c_coords = (int(point_c.x * w), int(point_c.y * h))
                
                # Calculate angle
                angle = self.calculate_angle(a_coords, b_coords, c_coords)
                angles[angle_name] = angle
                
        return angles

    def classify_exercise(self, frame, landmarks):
        """Classify the exercise using the neural network model."""
        # If model is not loaded, return "Unknown"
        if not self.model_loaded:
            return "Unknown", {}, self.get_joint_angles(landmarks, frame.shape)

        try:
            # Preprocess frame for neural network
            processed_tensor = self.preprocess_frame(frame).unsqueeze(0).to(self.device)
            # Get model prediction
            with torch.no_grad():
                output = self.model(processed_tensor)
            # Get exercise name with proper capitalization
            exercise = self.exercise_classes[torch.argmax(output, dim=1).item()] if torch.argmax(output, dim=1).item() < len(self.exercise_classes) else "Unknown"
            # Get joint angles
            angles = self.get_joint_angles(landmarks, frame.shape)
            return exercise, {}, angles
        except Exception as e:
            print(f"Exception in classify_exercise: {e}")
            return "Unknown", {}, self.get_joint_angles(landmarks, frame.shape)

    def generate_feedback(self, avg_angles, exercise_name, tolerance=10):
        """Generate detailed feedback based on average angles and ideal angles, but avoid mentioning left/right discrepancies due to camera angle."""
        feedback_messages = []
        ideal_angles = self.ideal_angles.get(exercise_name, {})
        # Group left/right pairs
        paired_joints = [('Left Elbow', 'Right Elbow'), ('Left Shoulder', 'Right Shoulder'),
                         ('Left Hip', 'Right Hip'), ('Left Knee', 'Right Knee')]
        used = set()
        for left, right in paired_joints:
            if left in ideal_angles and right in ideal_angles and left in avg_angles and right in avg_angles:
                avg_val = (avg_angles[left] + avg_angles[right]) / 2
                avg_ideal = (ideal_angles[left] + ideal_angles[right]) / 2
                diff = avg_val - avg_ideal
                if abs(diff) > tolerance:
                    suggestion = (f"Average {left.split()[1]}: Your average is {avg_val:.1f}° "
                                  f"(ideal {avg_ideal}°); diff = {abs(diff):.1f}°. ")
                    if diff > 0:
                        suggestion += "Consider reducing this angle."
                    else:
                        suggestion += "Consider increasing this angle."
                    feedback_messages.append(suggestion)
                else:
                    feedback_messages.append(f"Average {left.split()[1]}: Good! ({avg_val:.1f}° is close to ideal {avg_ideal}°)")
                used.add(left)
                used.add(right)
        # Handle unpaired or other angles
        for angle_name in ideal_angles:
            if angle_name in used:
                continue
            if angle_name in avg_angles:
                diff = avg_angles[angle_name] - ideal_angles[angle_name]
                if abs(diff) > tolerance:
                    suggestion = (f"{angle_name}: Your average is {avg_angles[angle_name]:.1f}° "
                                f"(ideal {ideal_angles[angle_name]}°); diff = {abs(diff):.1f}°. ")
                    if diff > 0:
                        suggestion += "Consider reducing this angle."
                    else:
                        suggestion += "Consider increasing this angle."
                    feedback_messages.append(suggestion)
                else:
                    feedback_messages.append(f"{angle_name}: Good! ({avg_angles[angle_name]:.1f}° is close to ideal {ideal_angles[angle_name]}°)")
            else:
                feedback_messages.append(f"{angle_name}: Data not available.")
        return feedback_messages

    def process_video(self, video_path):
        """Process a video file and return the path to the processed video, angle data, average angles, classified exercise, and feedback."""
        cap = None
        out = None
        output_path = None

        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Define the output path
            uploads_dir = Path(__file__).parent.parent.parent / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(uploads_dir / f"processed_{Path(video_path).name}")

            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"[PoseAnalyzer] Error: VideoWriter could not open output file: {output_path}")
                print(f"[PoseAnalyzer] FourCC: {fourcc}, FPS: {fps}, Dimensions: ({width}, {height})")
                raise IOError("VideoWriter could not open output file.")

            print(f"[PoseAnalyzer] Output video will be saved to: {output_path}")
            
            angle_data = []
            exercise_predictions = []
            all_angles_per_joint = {angle_name: [] for angle_name in self.ANGLES_TO_CALCULATE.keys()}
            
            smoother = LandmarkSmoother(window_size=5)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # Smooth landmarks
                    smoothed_landmarks = smoother.smooth(results.pose_landmarks.landmark)
                    # Get angles for this frame using smoothed landmarks
                    angles = self.get_joint_angles(smoothed_landmarks, frame.shape)
                    
                    # Store angles
                    angle_data.append(angles)
                    for joint, angle in angles.items():
                        all_angles_per_joint[joint].append(angle)
                    
                    # Classify exercise
                    exercise, _, _ = self.classify_exercise(frame, smoothed_landmarks)
                    exercise_predictions.append(exercise)
                    
                    # Draw smoothed landmarks and angles
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,  # For drawing, you can use original or smoothed
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # Display angles on frame
                    y_offset = 30
                    for joint, angle in angles.items():
                        cv2.putText(frame, f"{joint}: {angle:.1f}°",
                                  (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                  (0, 255, 0), 2, cv2.LINE_AA)
                        y_offset += 30
                
                # Write the frame to output video
                out.write(frame)
            
            # Calculate average angles
            avg_angles = {joint: np.mean(angles) if angles else 0 for joint, angles in all_angles_per_joint.items()}

            # Get the most common exercise prediction
            if exercise_predictions:
                counter = Counter(exercise_predictions)
                final_exercise = counter.most_common(1)[0][0]
            else:
                final_exercise = "Unknown"
            
            # Generate detailed feedback based on angles
            feedback = self.generate_feedback(avg_angles, final_exercise)
            
            return output_path, angle_data, avg_angles, final_exercise, feedback
        except Exception as e:
            print(f"[PoseAnalyzer] Error during video processing in PoseAnalyzer: {e}")
            raise e
        finally:
            # Ensure resources are released even if an error occurs
            if cap is not None and cap.isOpened():
                cap.release()
            if out is not None and out.isOpened():
                out.release()
            # Add a small delay to allow file handles to be released by the OS
            time.sleep(0.1) 