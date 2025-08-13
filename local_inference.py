"""
Simplified YOLO Duct Divider Tracker with Drone Position Data
Uses CSV log file with drone position to accurately identify sections
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time


@dataclass
class Section:
    """Represents a duct section"""
    id: int
    drone_position: float  # Position when detected
    frame_number: int
    confidence: float
    bbox: np.ndarray


class SimplifiedDuctTracker:
    def __init__(self, model_path: str, csv_path: str, section_length: float = 50.0, confidence_threshold: float = 0.5):
        """
        Initialize the tracker
        
        Args:
            model_path: Path to YOLO model
            csv_path: Path to CSV with drone position data
            section_length: Length of each duct section in cm
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.section_length = section_length
        
        # Load drone position data
        self.drone_data = self.load_drone_data(csv_path)
        
        # Tracking state
        self.sections: List[Section] = []
        self.frame_count = 0
        self.next_section_id = 1
        
        # Video timing
        self.video_start_time = 0  # Will be set when video starts
        self.fps = 30  # Will be updated from video
        
        # Position tracking
        self.last_section_position = None
        self.min_section_distance = section_length * 0.8  # 80% of section length as minimum
        
    def load_drone_data(self, csv_path: str) -> pd.DataFrame:
        """Load and prepare drone position data from CSV"""
        try:
            df = pd.read_csv(csv_path)
            print(f"📊 CSV columns: {df.columns.tolist()}")
            
            df['position'] = -df['x']*100
            df['position_y'] = df['y']
            df['position_z'] = df['z']
            
            # Convert timestamp to seconds from start
            if df['timestamp'].dtype == 'object':
                # Try to parse timestamp strings
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
                except:
                    print("⚠️  Could not parse timestamp. Using numeric values.")
                    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            
            # Ensure timestamps are relative to start (0-based)
            df['timestamp'] = df['timestamp'] - df['timestamp'].min()
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"📊 Loaded {len(df)} position records")
            print(f"📊 Time range: {df['timestamp'].min():.2f}s to {df['timestamp'].max():.2f}s")
            print(f"📊 Position range: {df['position'].min():.1f} to {df['position'].max():.1f}")
            
            return df[['timestamp', 'position']]
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            # Create dummy data if CSV fails
            return pd.DataFrame({'timestamp': [0], 'position': [0.0]})
    
    def get_drone_position(self, current_time: float) -> float:
        """Get drone position for a specific timestamp"""
        if self.drone_data.empty:
            return 0.0
        
        positions = self.drone_data['position']
        position_row = round(self.frame_count/2,0)

        position = positions[position_row]
            # Linear interpolation between closest timestamps
        print(f"🔍 position at {current_time:.2f}s: {position:.1f}cm")
        return position
    
    def is_new_section(self, current_position: float, bbox: np.ndarray) -> bool:
        """Determine if detection represents a new section based on drone position"""
        
        # If no sections yet, this is the first one
        if not self.sections:
            return True
        
        # Check distance from last detected section
        if self.last_section_position is not None:
            distance_traveled = abs(current_position - self.last_section_position)
            
            # New section if drone moved at least minimum distance
            if distance_traveled >= self.min_section_distance:
                return True
        
        # Check against all existing sections
        for section in self.sections:
            position_diff = abs(current_position - section.drone_position)
            if position_diff < self.min_section_distance:
                return False  # Too close to existing section
        
        return True
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """Process a single frame"""
        self.frame_count += 1
        
        # Calculate current video time in seconds
        current_time = (self.frame_count - 1) / self.fps
        
        # Get drone position for this timestamp
        drone_position = self.get_drone_position(current_time)
        
        # Run YOLO detection
        results = self.model.predict(frame, conf=self.confidence_threshold)
        annotated_frame = results[0].plot()
        
        current_detections = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Process the most confident detection
            best_idx = np.argmax(confidences)
            bbox = boxes[best_idx]
            conf = confidences[best_idx]
            
            detection_info = {
                'bbox': bbox,
                'confidence': conf,
                'drone_position': drone_position,
                'frame': self.frame_count,
                'time': current_time
            }
            current_detections.append(detection_info)
            
            # Check if this represents a new section
            if self.is_new_section(drone_position, bbox):
                new_section = Section(
                    id=self.next_section_id,
                    drone_position=drone_position,
                    frame_number=self.frame_count,
                    confidence=conf,
                    bbox=bbox
                )
                
                self.sections.append(new_section)
                self.last_section_position = drone_position
                self.next_section_id += 1
                
                print(f"✅ New section #{new_section.id} detected at position {drone_position:.1f}cm (frame {self.frame_count}, time {current_time:.2f}s)")
        
        # Add visualization
        self.add_info_overlay(annotated_frame, current_detections, drone_position, current_time)
        
        return annotated_frame, current_detections
    
    def add_info_overlay(self, frame: np.ndarray, detections: List[dict], drone_position: float, current_time: float):
        """Add information overlay to the frame"""
        
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Time: {current_time:.2f}s",
            f"Drone Position: {drone_position:.1f}cm",
            f"Current Detections: {len(detections)}",
            f"Sections Found: {len(self.sections)}",
            f"Section Length: {self.section_length:.1f}cm"
        ]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, 35 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show detection info
        for detection in detections:
            bbox = detection['bbox'].astype(int)
            conf = detection['confidence']
            
            cv2.putText(frame, f"Conf: {conf:.2f}", 
                       (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Show sections summary
        if self.sections:
            positions = sorted([s.drone_position for s in self.sections])
            summary = f"Sections at: {', '.join([f'{p:.0f}cm' for p in positions])}"
            cv2.putText(frame, summary, (20, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def run_video_tracking(self, video_path: str, output_path: str = None, display: bool = True):
        """Run tracking on video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video: {video_path}")
            return
        
        # Get video properties
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / self.fps
        
        print(f"📹 Video: {width}x{height} @ {self.fps}fps, {total_frames} frames ({video_duration:.1f}s)")
        
        # Check if we have enough log data
        if not self.drone_data.empty:
            log_duration = self.drone_data['timestamp'].max()
            print(f"📊 Log duration: {log_duration:.1f}s")
            if log_duration < video_duration * 0.8:
                print("⚠️  Warning: Log duration is much shorter than video duration")
        
        # Setup output video
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        print("🚀 Starting tracking... Press 'q' to quit")
        
        while True:
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
                cv2.imshow('Simplified Duct Tracker', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
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
        print(f"   Sections detected: {len(self.sections)}")
        
        if self.sections:
            print(f"\n📏 Section Positions:")
            sorted_sections = sorted(self.sections, key=lambda s: s.drone_position)
            
            for section in sorted_sections:
                print(f"   Section #{section.id}: {section.drone_position:.1f}cm "
                      f"(frame {section.frame_number})")
            
            # Calculate spacing
            if len(sorted_sections) > 1:
                spacings = []
                for i in range(len(sorted_sections) - 1):
                    spacing = abs(sorted_sections[i+1].drone_position - sorted_sections[i].drone_position)
                    spacings.append(spacing)
                
                avg_spacing = np.mean(spacings)
                print(f"\n   Average spacing: {avg_spacing:.1f}cm")
                print(f"   Expected spacing: {self.section_length:.1f}cm")
                print(f"   Spacing accuracy: {(1 - abs(avg_spacing - self.section_length) / self.section_length) * 100:.1f}%")
        else:
            print("   ⚠️  No sections detected. Check CSV data and detection parameters.")


def main():
    parser = argparse.ArgumentParser(description='Simplified Duct Tracker with Drone Position')
    parser.add_argument('--model', '-m', required=True, help='Path to YOLO model')
    parser.add_argument('--video', '-v', required=True, help='Path to video file')
    parser.add_argument('--csv', '-c', required=True, help='Path to CSV with drone position data')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--section-length', '-s', type=float, default=50.0, help='Section length in cm')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = SimplifiedDuctTracker(
        model_path=args.model,
        csv_path=args.csv,
        section_length=args.section_length,
        confidence_threshold=args.confidence
    )
    
    # Run tracking
    tracker.run_video_tracking(args.video, args.output, not args.no_display)


if __name__ == "__main__":
    main()