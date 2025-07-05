"""
YOLO Duct Segmentation - Real-time Local Inference
Run this script locally on your PC to test the trained model with video files
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
import os
from pathlib import Path
from collections import defaultdict, deque
import json

class DuctDetectorLocal:
    def __init__(self, model_path, confidence_threshold=0.5):
        """
        Initialize the duct detector for local inference with tracking
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Load the trained model
        print(f"🔄 Loading model from: {model_path}")
        self.model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.avg_fps = 0
        
        # NEW: Tracking components
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.session_positions = {}  # 3D positions of sessions
        self.drone_path = []  # Camera/drone trajectory
        self.camera_poses = [np.eye(4)]  # Camera pose history

        self.tube_width = 0.50  # 50cm in meters
        self.tube_height = 0.50  # 50cm in meters
        
        # Camera calibration (will be updated based on actual frame size)
        self.camera_matrix = None
        self.frame_width = None
        self.frame_height = None
        
        # Tracking improvements
        self.min_track_length = 3  # Minimum frames to confirm a track
        self.max_disappeared = 10  # Max frames a track can disappear
        self.track_confidence_threshold = 0.3  # Lower for small objects

        
        
    def set_camera_calibration(self, frame_width, frame_height):
        """Set camera calibration based on actual frame dimensions"""
        # Estimate focal length as ~0.8 * frame_width (common approximation)
        fx = fy = 0.8 * frame_width
        cx = frame_width / 2
        cy = frame_height / 2
        
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        print(f"📐 Camera calibration set for {frame_width}x{frame_height}")
        print(f"   Focal length: {fx:.1f}")
        print(f"   Principal point: ({cx:.1f}, {cy:.1f})")
        
    def process_frame(self, frame):
        """
        Process a single frame and return annotated result
        """
        # Run inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Get the annotated frame
        annotated_frame = results[0].plot()
        
        # Extract detection information
        detections = []
        if results[0].masks is not None:
            for i, (box, mask, conf, cls) in enumerate(zip(
                results[0].boxes.xyxy.cpu().numpy(),
                results[0].masks.xy,
                results[0].boxes.conf.cpu().numpy(),
                results[0].boxes.cls.cpu().numpy()
            )):
                detection = {
                    'bbox': box,
                    'mask': mask,
                    'confidence': conf,
                    'class': int(cls),
                    'class_name': self.model.names[int(cls)]
                }
                detections.append(detection)
        
        return annotated_frame, detections

    def process_frame_with_tracking(self, frame, frame_idx):
        """
        Enhanced tracking with validation
        """
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Set camera calibration if not done yet
        if self.camera_matrix is None:
            self.set_camera_calibration(frame_width, frame_height)
            self.frame_width = frame_width
            self.frame_height = frame_height
        
        # Run inference with tracking
        results = self.model.track(
            frame, 
            conf=self.confidence_threshold,
            persist=True, 
            tracker="bytetrack.yaml",
            iou=0.3,
            max_det=20,
            agnostic_nms=True
        )
        
        annotated_frame = results[0].plot()
        tracked_detections = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                bbox_area = (box[2] - box[0]) * (box[3] - box[1])
                
                # Store tracking history
                self.track_history[track_id].append({
                    'frame': frame_idx,
                    'center': center,
                    'bbox': box,
                    'bbox_area': bbox_area,
                    'confidence': conf,
                    'class': int(cls)
                })
                
                tracked_detections.append({
                    'track_id': track_id,
                    'bbox': box,
                    'center': center,
                    'bbox_area': bbox_area,
                    'confidence': conf,
                    'class': int(cls),
                    'class_name': self.model.names[int(cls)]
                })
        
        # Validate and filter tracks
        valid_tracks = self.validate_and_filter_tracks(tracked_detections, frame_idx)
        
        return annotated_frame, valid_tracks

    def validate_and_filter_tracks(self, tracked_detections, frame_idx):
        """
        Filter out bad tracks and validate session positions
        """
        valid_tracks = []
        
        for detection in tracked_detections:
            track_id = detection['track_id']
            
            # Check track history length
            if len(self.track_history[track_id]) < self.min_track_length:
                continue
            
            # Check if track is stable (not jumping around too much)
            recent_positions = [item['center'] for item in list(self.track_history[track_id])[-5:]]
            if len(recent_positions) >= 2:
                # Calculate movement variance
                positions = np.array(recent_positions)
                movement_variance = np.var(positions, axis=0)
                
                # Skip tracks that move too erratically
                if np.max(movement_variance) > 5000:  # Adjust threshold
                    continue
            
            # Check confidence trend
            recent_confidences = [item['confidence'] for item in list(self.track_history[track_id])[-3:]]
            avg_confidence = np.mean(recent_confidences)
            
            if avg_confidence < self.track_confidence_threshold:
                continue
            
            valid_tracks.append(detection)
        
        return valid_tracks
    
    def estimate_depth_from_size(self, bbox_area, frame_width, frame_height):
        """
        Improved depth estimation for 50x50cm tube
        """
        # Calculate what percentage of frame the bbox occupies
        frame_area = frame_width * frame_height
        bbox_ratio = bbox_area / frame_area
        
        print(f"Debug: bbox_area={bbox_area}, frame_area={frame_area}, ratio={bbox_ratio:.4f}")
        
        # For a 50cm tube, calibrate based on your actual footage
        if bbox_ratio > 0.15:  # Very close
            return 0.2  # 20cm from camera
        elif bbox_ratio > 0.08:  # Medium distance
            return 0.5  # 50cm from camera
        elif bbox_ratio > 0.03:  # Far
            return 1.0  # 1m from camera
        else:  # Very far
            return 2.0  # 2m from camera
    
    def pixel_to_world(self, center_2d, depth, camera_pose, frame_width, frame_height):
        """
        Improved 3D conversion for tube environment
        """
        # Use actual camera matrix
        fx, fy = self.camera_matrix[0,0], self.camera_matrix[1,1]
        cx, cy = self.camera_matrix[0,2], self.camera_matrix[1,2]
        
        # Convert to camera coordinates
        x_cam = (center_2d[0] - cx) * depth / fx
        y_cam = (center_2d[1] - cy) * depth / fy
        z_cam = depth
        
        # Constrain to tube dimensions (50x50cm)
        x_cam = np.clip(x_cam, -0.25, 0.25)  # ±25cm from center
        y_cam = np.clip(y_cam, -0.25, 0.25)  # ±25cm from center
        
        # Convert to world coordinates
        cam_point = np.array([x_cam, y_cam, z_cam, 1])
        world_point = camera_pose @ cam_point
        
        return world_point[:3]
    
    def estimate_3d_positions(self, tracked_detections, camera_pose, frame_width, frame_height):
        """
        Improved 3D estimation for tube environment
        """
        for detection in tracked_detections:
            track_id = detection['track_id']
            center_2d = detection['center']
            bbox_area = detection['bbox_area']
            
            # Better depth estimation
            estimated_depth = self.estimate_depth_from_size(bbox_area, frame_width, frame_height)
            
            # Convert to 3D
            world_pos = self.pixel_to_world(center_2d, estimated_depth, camera_pose, 
                                        frame_width, frame_height)
            
            # More aggressive smoothing for small tube
            if track_id in self.session_positions:
                prev_pos = self.session_positions[track_id]
                alpha = 0.1  # More smoothing
                self.session_positions[track_id] = (
                    alpha * world_pos[0] + (1-alpha) * prev_pos[0],
                    alpha * world_pos[1] + (1-alpha) * prev_pos[1],
                    alpha * world_pos[2] + (1-alpha) * prev_pos[2]
                )
            else:
                self.session_positions[track_id] = world_pos

    def estimate_camera_motion(self, frame, prev_frame):
        """Estimate camera motion using feature matching"""
        if prev_frame is None:
            return np.eye(4)
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect and match features
        detector = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)
        
        if des1 is not None and des2 is not None and len(des1) > 10:
            # Match features
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) > 20:
                # Extract matched points
                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
                
                # Estimate homography
                H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
                
                if H is not None:
                    # Convert homography to simple translation
                    translation = np.array([H[0, 2], H[1, 2], 0.1])
                    
                    # Create transformation matrix
                    transform = np.eye(4)
                    transform[:3, 3] = translation * 0.01
                    return transform
        
        # Default: assume small forward motion
        transform = np.eye(4)
        transform[2, 3] = 0.01
        return transform
    
    def draw_tracking_info(self, frame, tracked_detections):
        """Draw tracking information on frame"""
        for detection in tracked_detections:
            track_id = detection['track_id']
            center = detection['center']
            
            # Draw track ID
            cv2.putText(frame, f"ID:{track_id}", 
                       (int(center[0]), int(center[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw tracking trail
            if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                points = [item['center'] for item in list(self.track_history[track_id])[-10:]]
                for i in range(1, len(points)):
                    cv2.line(frame, 
                           (int(points[i-1][0]), int(points[i-1][1])),
                           (int(points[i][0]), int(points[i][1])),
                           (0, 255, 255), 2)

    def run_video_inference_with_tracking(self, video_path, output_path=None, 
                                        reconstruction_path=None, display=True):
        """Run inference with tracking and 3D reconstruction"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video Info (with tracking):")
        print(f"   Resolution: {frame_width}x{frame_height}")
        print(f"   FPS: {fps}")
        print(f"   Total Frames: {total_frames}")
        
        # Setup video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            print(f"💾 Output will be saved to: {output_path}")
        
        # Processing loop
        frame_count = 0
        prev_frame = None
        
        print("\n🚀 Starting tracking inference...")
        print("Press 'q' to quit, 'p' to pause/resume, 's' to save screenshot")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n✅ Video processing completed!")
                    break
                
                frame_count += 1
                
                # Estimate camera motion
                if prev_frame is not None:
                    relative_pose = self.estimate_camera_motion(frame, prev_frame)
                    current_pose = self.camera_poses[-1] @ relative_pose
                    self.camera_poses.append(current_pose)
                    self.drone_path.append(current_pose[:3, 3])
                
                # Process frame with tracking
                start_time = time.time()
                annotated_frame, tracked_detections = self.process_frame_with_tracking(
                    frame, frame_count)
                processing_time = time.time() - start_time
                
                # Estimate 3D positions
                if tracked_detections:
                    self.estimate_3d_positions(tracked_detections, self.camera_poses[-1], 
                                             frame_width, frame_height)
                
                # Draw tracking info
                self.draw_tracking_info(annotated_frame, tracked_detections)
                
                # Calculate FPS
                current_fps = self.calculate_fps()
                
                # Add info to frame
                info_text = [
                    f"Frame: {frame_count}/{total_frames} (TRACKING)",
                    f"FPS: {current_fps:.1f}",
                    f"Processing: {processing_time*1000:.1f}ms",
                    f"Tracked Objects: {len(tracked_detections)}",
                    f"Total Sessions: {len(self.session_positions)}"
                ]
                
                y_offset = 30
                for text in info_text:
                    cv2.putText(annotated_frame, text, (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25
                
                # Save and display
                if out:
                    out.write(annotated_frame)
                
                if display:
                    cv2.imshow('YOLO Duct Detection - Tracking', annotated_frame)
                
                prev_frame = frame.copy()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Stopping inference...")
                break
            elif key == ord('p'):
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"⏸️ {status}")
            elif key == ord('s'):
                screenshot_path = f"tracking_screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Save reconstruction data
        if reconstruction_path:
            self.save_reconstruction(reconstruction_path)
        
        print(f"\n📊 Tracking Results:")
        print(f"   Unique Sessions Tracked: {len(self.session_positions)}")
        print(f"   Drone Path Points: {len(self.drone_path)}")

    def save_reconstruction(self, output_file):
        """Save 3D reconstruction data"""
        # Custom JSON encoder for numpy arrays
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, deque):
                    return list(obj)
                return super().default(obj)
        
        reconstruction_data = {
            'session_positions': {str(k): v for k, v in self.session_positions.items()},
            'drone_path': self.drone_path,
            'track_history': {str(k): list(v) for k, v in self.track_history.items()},
            'camera_poses': self.camera_poses
        }
        
        with open(output_file, 'w') as f:
            json.dump(reconstruction_data, f, indent=2, cls=NumpyEncoder)
        
        print(f"💾 3D reconstruction saved to: {output_file}")
    
    def calculate_fps(self):
        """Calculate and return current FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.avg_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
        
        return self.avg_fps
    
    # Keep your existing run_video_inference and run_webcam_inference methods...
    def run_video_inference(self, video_path, output_path=None, display=True):
        # ... your existing code ...
        pass
    
    def run_webcam_inference(self, camera_id=0):
        # ... your existing code ...
        pass

def main():
    parser = argparse.ArgumentParser(description='YOLO Duct Detection - Local Inference')
    parser.add_argument('--model', '-m', required=True, help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--video', '-v', help='Path to input video file')
    parser.add_argument('--camera', '-c', type=int, help='Camera device ID (default: 0)')
    parser.add_argument('--output', '-o', help='Path to save output video')
    parser.add_argument('--confidence', '-conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--no-display', action='store_true', help='Disable real-time display')
    parser.add_argument('--tracking', '-t', action='store_true', help='Enable object tracking')
    parser.add_argument('--reconstruction', '-r', help='Path to save 3D reconstruction data (JSON)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    # Initialize detector
    detector = DuctDetectorLocal(args.model, args.confidence)
    
    # Run inference
    if args.video:
        if not os.path.exists(args.video):
            print(f"❌ Error: Video file not found: {args.video}")
            return
        if args.tracking:
            detector.run_video_inference_with_tracking(
                args.video, args.output, args.reconstruction, not args.no_display)
        else:
            detector.run_video_inference(args.video, args.output, not args.no_display)
    elif args.camera is not None:
        detector.run_webcam_inference(args.camera)
    else:
        print("❌ Error: Please specify either --video or --camera")

if __name__ == "__main__":
    main()