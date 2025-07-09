"""
Improved YOLO Duct Divider Tracker
Enhanced motion analysis and distance-based tracking for unique dividers
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import math


@dataclass
class Divider:
    """Represents a unique divider"""
    id: int
    estimated_distance: float
    first_seen_frame: int
    last_seen_frame: int
    confidence: float
    bbox: np.ndarray
    size_history: List[float]
    

class DuctTracker:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """Initialize the tracker"""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Tracking parameters
        self.reference_divider_size = 50  # cm - known real divider size
        self.focal_length = 800  # pixels - camera focal length estimate
        self.min_divider_separation = 10  # cm - minimum distance between unique dividers
        self.size_change_threshold = 0.01  # 15% size change to indicate motion
        
        # State tracking
        self.unique_dividers: List[Divider] = []
        self.next_divider_id = 1
        self.frame_count = 0
        
        # Motion analysis
        self.prev_frame_gray = None
        self.prev_detection_size = None
        self.size_history = deque(maxlen=10)
        self.distance_history = deque(maxlen=15)
        
        # Detection stability
        self.detection_buffer = deque(maxlen=5)
        self.no_detection_frames = 0
        
        # Movement direction estimation
        self.movement_direction = 0  # 1 = forward, -1 = backward, 0 = stationary
        self.direction_history = deque(maxlen=8)
        
    def estimate_distance(self, bbox: np.ndarray) -> float:
        """Estimate distance based on divider apparent size"""
        x1, y1, x2, y2 = bbox
        apparent_size = max(x2 - x1, y2 - y1)  # Use larger dimension
        
        # Distance = (Real_Size * Focal_Length) / Apparent_Size
        distance = (self.reference_divider_size * self.focal_length) / max(apparent_size, 1)
        return distance
    
    def calculate_detection_size(self, bbox: np.ndarray) -> float:
        """Calculate the size of the detection box"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return max(width, height)  # Use the larger dimension
    
    def estimate_movement_direction(self, current_distance: float) -> int:
        """Estimate movement direction based on distance changes"""
        if len(self.distance_history) < 3:
            return 0
        
        recent_distances = list(self.distance_history)[-5:]
        if len(recent_distances) < 3:
            return 0
        
        # Calculate trend
        x = np.arange(len(recent_distances))
        y = np.array(recent_distances)
        
        # Simple linear regression to detect trend
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            
            # Threshold for significant movement
            if abs(slope) > 0.5:  # 2cm per frame threshold
                return -1 if slope > 0 else 1  # Negative distance change = moving forward
        
        return 0
    
    def detect_new_divider_approach(self, current_distance: float, current_size: float) -> bool:
        """Detect if we're approaching a new divider"""
        
        # Need some history to compare
        if len(self.distance_history) < 3:
            return False
        
        recent_distances = list(self.distance_history)[-5:]
        
        # Check for rapid decrease in distance (approaching)
        distance_change = recent_distances[-1] - recent_distances[0]
        
        # If distance is decreasing rapidly, we might be approaching
        if distance_change < -1:  # 20cm decrease over 5 frames
            return True
        
        # Check for size increase (getting closer)
        if len(self.size_history) >=1:
            size_change = (current_size - self.size_history[0]) / self.size_history[0]
            if size_change > self.size_change_threshold:
                return True
        
        return False
    
    def is_new_divider(self, distance: float, bbox: np.ndarray) -> bool:
        """Enhanced logic to determine if detection represents a new unique divider"""
        
        current_size = self.calculate_detection_size(bbox)
        
        # Store current detection info
        self.distance_history.append(distance)
        self.size_history.append(current_size)
        
        # Update movement direction
        self.movement_direction = self.estimate_movement_direction(distance)
        self.direction_history.append(self.movement_direction)
        
        # If no existing dividers, this is the first one
        if not self.unique_dividers:
            return True
        
        # Check distance separation from existing dividers
        for divider in self.unique_dividers:
            distance_diff = abs(distance - divider.estimated_distance)
            
            # If very close to existing divider, update it
            if distance_diff < self.min_divider_separation:
                divider.last_seen_frame = self.frame_count
                divider.size_history.append(current_size)
                return False
        
        # Check for approach pattern
        if self.detect_new_divider_approach(distance, current_size):
            # Additional validation: check if we've passed through a gap
            if len(self.distance_history) >= 10:
                # Look for a pattern where distance increased then decreased
                mid_point = len(self.distance_history) // 2
                early_distances = list(self.distance_history)[:mid_point]
                recent_distances = list(self.distance_history)[mid_point:]
                
                if early_distances and recent_distances:
                    early_avg = np.mean(early_distances)
                    recent_avg = np.mean(recent_distances)
                    
                    # If we had larger distances before and now smaller, we passed through
                    if early_avg > recent_avg + 30:  # 30cm threshold
                        return True
        
        # Check for consistent movement in one direction
        if len(self.direction_history) >= 5:
            recent_directions = list(self.direction_history)[-5:]
            if abs(sum(recent_directions)) >= 3:  # Consistent movement
                
                # Check if current distance is significantly different from last divider
                if self.unique_dividers:
                    last_divider = self.unique_dividers[-1]
                    frames_since_last = self.frame_count - last_divider.last_seen_frame
                    
                    # If enough time has passed and distance is different
                    if frames_since_last > 20 and abs(distance - last_divider.estimated_distance) > self.min_divider_separation:
                        return True
        
        return False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """Process a single frame"""
        self.frame_count += 1
        
        # Run YOLO detection
        results = self.model.predict(frame, conf=self.confidence_threshold)
        annotated_frame = results[0].plot()
        
        current_detections = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            self.no_detection_frames = 0
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Process the most confident detection
            best_idx = np.argmax(confidences)
            bbox = boxes[best_idx]
            conf = confidences[best_idx]
            
            # Estimate distance
            distance = self.estimate_distance(bbox)
            
            detection_info = {
                'bbox': bbox,
                'confidence': conf,
                'distance': distance,
                'size': self.calculate_detection_size(bbox)
            }
            current_detections.append(detection_info)
            
            # Check if this is a new unique divider
            if self.is_new_divider(distance, bbox):
                new_divider = Divider(
                    id=self.next_divider_id,
                    estimated_distance=distance,
                    first_seen_frame=self.frame_count,
                    last_seen_frame=self.frame_count,
                    confidence=conf,
                    bbox=bbox,
                    size_history=[self.calculate_detection_size(bbox)]
                )
                
                self.unique_dividers.append(new_divider)
                self.next_divider_id += 1
                
                print(f"✅ New divider #{new_divider.id} detected at {distance:.1f}cm (frame {self.frame_count})")
        
        else:
            self.no_detection_frames += 1
        
        # Add visualization
        self.add_info_overlay(annotated_frame, current_detections)
        
        return annotated_frame, current_detections
    
    def add_info_overlay(self, frame: np.ndarray, detections: List[dict]):
        """Add information overlay to the frame"""
        
        # Create info panel
        movement_dir_str = {1: "Forward", -1: "Backward", 0: "Stationary"}[self.movement_direction]
        
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Movement: {movement_dir_str}",
            f"Current Detections: {len(detections)}",
            f"Unique Dividers: {len(self.unique_dividers)}",
            f"No Detection: {self.no_detection_frames} frames",
            f"Confidence: {self.confidence_threshold:.2f}"
        ]
        
        # Add semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, 35 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show current detections with distances
        for detection in detections:
            bbox = detection['bbox'].astype(int)
            distance = detection['distance']
            conf = detection['confidence']
            
            # Draw distance info near detection
            cv2.putText(frame, f"{distance:.0f}cm ({conf:.2f})", 
                       (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Show unique dividers summary
        if self.unique_dividers:
            distances = sorted([d.estimated_distance for d in self.unique_dividers])
            summary = f"Dividers: {', '.join([f'{d:.0f}cm' for d in distances])}"
            cv2.putText(frame, summary, (20, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show distance history as a simple graph
        if len(self.distance_history) > 1:
            history_display = f"Distance trend: {list(self.distance_history)[-5:]}"
            cv2.putText(frame, f"Recent distances: {[f'{d:.0f}' for d in list(self.distance_history)[-5:]]}", 
                       (20, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def run_video_tracking(self, video_path: str, output_path: str = None, display: bool = True):
        """Run tracking on video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup output video
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("🚀 Starting tracking... Press 'q' to quit, 'p' to pause")
        
        paused = False
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame, detections = self.process_frame(frame)
                
                # Save output
                if out:
                    out.write(annotated_frame)
                
                # Display
                if display:
                    cv2.imshow('Improved Duct Divider Tracker', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        paused = not paused
            else:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = False
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Print results
        self.print_results()
    
    def print_results(self):
        """Print final tracking results"""
        print(f"\n📊 Tracking Results:")
        print(f"   Total frames processed: {self.frame_count}")
        print(f"   Unique dividers found: {len(self.unique_dividers)}")
        
        if self.unique_dividers:
            print(f"\n📏 Divider Positions:")
            sorted_dividers = sorted(self.unique_dividers, key=lambda d: d.estimated_distance)
            
            for divider in sorted_dividers:
                frame_duration = divider.last_seen_frame - divider.first_seen_frame + 1
                print(f"   Divider #{divider.id}: {divider.estimated_distance:.1f}cm "
                      f"(frames {divider.first_seen_frame}-{divider.last_seen_frame}, "
                      f"duration: {frame_duration} frames)")
            
            # Calculate spacing
            if len(sorted_dividers) > 1:
                spacings = []
                for i in range(len(sorted_dividers) - 1):
                    spacing = abs(sorted_dividers[i+1].estimated_distance - sorted_dividers[i].estimated_distance)
                    spacings.append(spacing)
                
                avg_spacing = np.mean(spacings)
                print(f"\n   Average spacing: {avg_spacing:.1f}cm")
                print(f"   Spacing range: {min(spacings):.1f}cm - {max(spacings):.1f}cm")
        else:
            print("   ⚠️  No unique dividers detected. Consider adjusting parameters.")


def main():
    parser = argparse.ArgumentParser(description='Improved Duct Divider Tracker')
    parser.add_argument('--model', '-m', required=True, help='Path to YOLO model')
    parser.add_argument('--video', '-v', required=True, help='Path to video file')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = DuctTracker(args.model, args.confidence)
    
    # Run tracking
    tracker.run_video_tracking(args.video, args.output, not args.no_display)


if __name__ == "__main__":
    main()