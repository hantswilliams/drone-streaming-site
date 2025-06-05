#!/usr/bin/env python3
"""
YOLO v8 RTMP Stream and Video File Analysis
Real-time object detection and segmentation using YOLOv8 on RTMP streams or local video files
"""

import cv2
import numpy as np
import time
import threading
import queue
import argparse
import sys
import os
from datetime import datetime
import json
from ultralytics import YOLO
import torch

class YOLOv8StreamProcessor:
    """Real-time processor with YOLOv8 object detection and segmentation for RTMP streams or local video files"""

    def __init__(self, input_source="rtmp://34.67.35.85:1935/live/livestream", output_dir="./yolo_output", is_video_file=False):
        self.input_source = input_source
        self.output_dir = output_dir
        self.is_video_file = is_video_file
        self.cap = None
        self.running = False
        self.capture_finished = False  # Flag to track when video capture is done
        self.frame_queue = queue.Queue(maxsize=10)
        self.detection_results = []
        
        # YOLOv8 Configuration - LOCKED TO SMALL MODEL ONLY
        self.model_name = "yolov8s.pt"  # Small model - only this model will be downloaded
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.enable_segmentation = True
        self.enable_tracking = True
        
        # Initialize YOLO model (will only download yolov8s.pt on first run)
        try:
            print(f"Loading YOLOv8 Small Model: {self.model_name}")
            print("Note: Only the small model (yolov8s.pt) will be downloaded to save space")
            self.model = YOLO(self.model_name)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {self.device}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            sys.exit(1)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
    def connect_to_source(self):
        """Connect to input source (RTMP stream or local video file)"""
        if self.is_video_file:
            print(f"Opening local video file: {self.input_source}")
            if not os.path.exists(self.input_source):
                print(f"Error: Video file not found: {self.input_source}")
                return False
        else:
            print(f"Connecting to RTMP stream: {self.input_source}")
        
        # Configure OpenCV for RTMP or video file
        self.cap = cv2.VideoCapture(self.input_source)
        
        # Set buffer size to reduce latency (mainly for streams)
        if not self.is_video_file:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            source_type = "video file" if self.is_video_file else "RTMP stream"
            print(f"Error: Could not open {source_type}")
            return False
            
        # Get source properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        source_type = "Video file" if self.is_video_file else "Stream"
        print(f"{source_type} properties: {width}x{height} @ {fps} FPS")
        
        # For video files, also show duration and frame count
        if self.is_video_file:
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            print(f"Video duration: {duration:.2f} seconds ({frame_count} frames)")
        
        return True
    
    def process_frame_with_yolo(self, frame):
        """Process frame with YOLOv8 detection and segmentation"""
        try:
            # Run YOLOv8 inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            detections = []
            annotated_frame = frame.copy()
            
            for result in results:
                # Get detection boxes
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = box.astype(int)
                        class_name = self.model.names[class_id]
                        
                        # Store detection info
                        detection = {
                            'class_id': int(class_id),
                            'class_name': class_name,
                            'confidence': float(conf),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'timestamp': datetime.now().isoformat()
                        }
                        detections.append(detection)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Handle segmentation masks if available
                if self.enable_segmentation and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    for i, mask in enumerate(masks):
                        # Resize mask to frame size
                        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        
                        # Create colored overlay
                        color = np.random.randint(0, 255, 3).tolist()
                        colored_mask = np.zeros_like(annotated_frame)
                        colored_mask[mask_resized > 0.5] = color
                        
                        # Blend with frame
                        annotated_frame = cv2.addWeighted(annotated_frame, 0.8, colored_mask, 0.2, 0)
            
            return annotated_frame, detections
            
        except Exception as e:
            print(f"Error in YOLO processing: {e}")
            return frame, []
    
    def save_detection_results(self, detections, frame_number):
        """Save detection results to JSON file"""
        if detections:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"detections_{timestamp}_frame_{frame_number}.json")
            
            with open(filename, 'w') as f:
                json.dump({
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': detections,
                    'total_objects': len(detections)
                }, f, indent=2)
    
    def save_frame(self, frame, frame_number, prefix="detection"):
        """Save annotated frame to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"{prefix}_{timestamp}_{frame_number}.jpg")
        cv2.imwrite(filename, frame)
        return filename
    
    def calculate_fps(self):
        """Calculate and display FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            fps = self.fps_counter / (current_time - self.fps_start_time)
            print(f"Processing FPS: {fps:.2f}")
            self.fps_counter = 0
            self.fps_start_time = current_time
            return fps
        return None
    
    def frame_capture_worker(self):
        """Worker thread for capturing frames"""
        frames_read_successfully = 0
        consecutive_failures = 0
        
        while self.running:
            if self.cap is not None:
                ret, frame = self.cap.read()
                if ret:
                    frames_read_successfully += 1
                    consecutive_failures = 0
                    
                    # For video files, wait until queue has space (don't drop frames)
                    # For streams, drop frames to maintain real-time processing
                    if self.is_video_file:
                        # Wait for queue space - don't drop frames for video files
                        while self.frame_queue.full() and self.running:
                            time.sleep(0.01)
                        if self.running:
                            self.frame_queue.put(frame)
                    else:
                        # For streams, drop frames if queue is full (reduce latency)
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame)
                        else:
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put(frame)
                            except queue.Empty:
                                pass
                else:
                    consecutive_failures += 1
                    
                    if self.is_video_file:
                        if consecutive_failures >= 3:
                            print(f"Reached end of video file (read {frames_read_successfully} frames successfully)")
                            self.capture_finished = True
                            break
                        else:
                            # Might be a temporary read issue, try a few more times
                            time.sleep(0.01)
                    else:
                        print("Failed to read frame from stream")
                        time.sleep(0.1)
            else:
                time.sleep(0.1)
    
    def start_processing(self):
        """Start the input source processing"""
        if not self.connect_to_source():
            return False
        
        self.running = True
        self.start_time = time.time()
        
        # Start frame capture thread
        capture_thread = threading.Thread(target=self.frame_capture_worker)
        capture_thread.daemon = True
        capture_thread.start()
        
        source_type = "video file" if self.is_video_file else "RTMP stream"
        print(f"Starting YOLOv8 {source_type} processing...")
        print("Press 'q' to quit, 's' to save current frame")
        
        try:
            empty_queue_count = 0
            max_empty_queue_attempts = 10  # Allow some attempts for empty queue
            
            while self.running:
                try:
                    # Get frame from queue
                    frame = self.frame_queue.get(timeout=1.0)
                    self.frame_count += 1
                    empty_queue_count = 0  # Reset counter when we get a frame
                    
                    # Process frame with YOLOv8
                    annotated_frame, detections = self.process_frame_with_yolo(frame)
                    
                    if detections:
                        self.detection_count += len(detections)
                        print(f"Frame {self.frame_count}: Found {len(detections)} objects")
                        
                        # Save detection results
                        self.save_detection_results(detections, self.frame_count)
                        
                        # Save frame with detections
                        self.save_frame(annotated_frame, self.frame_count)
                    else:
                        print(f"Frame {self.frame_count}: No objects detected")
                    
                    # Calculate FPS
                    self.calculate_fps()
                    
                    # Display frame
                    cv2.imshow(f'YOLOv8 {source_type.title()} Analysis', annotated_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quit requested by user")
                        break
                    elif key == ord('s'):
                        filename = self.save_frame(annotated_frame, self.frame_count, "manual_save")
                        print(f"Frame saved: {filename}")
                        
                except queue.Empty:
                    empty_queue_count += 1
                    
                    if self.is_video_file:
                        # For video files, check if capture is finished and queue is empty
                        if self.capture_finished and self.frame_queue.empty():
                            print("Video file processing completed - all frames processed")
                            break
                        elif empty_queue_count >= max_empty_queue_attempts:
                            print("Video file processing completed - timeout waiting for frames")
                            break
                        else:
                            print(f"Waiting for more frames... ({empty_queue_count}/{max_empty_queue_attempts})")
                            time.sleep(0.1)
                            continue
                    else:
                        print("No frames available, checking stream...")
                        continue
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Save final statistics
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        stats = {
            'total_frames_processed': self.frame_count,
            'total_detections': self.detection_count,
            'duration_seconds': duration,
            'average_fps': self.frame_count / duration if duration > 0 else 0,
            'detections_per_minute': (self.detection_count / duration * 60) if duration > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        stats_file = os.path.join(self.output_dir, f"analysis_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Processing completed:")
        print(f"  Frames processed: {self.frame_count}")
        print(f"  Total detections: {self.detection_count}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Average FPS: {stats['average_fps']:.2f}")
        print(f"  Stats saved to: {stats_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="YOLOv8 Small Model Analysis (RTMP Stream or Local Video)")
    
    # Input source - mutually exclusive group
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--url", default="rtmp://34.67.35.85:1935/live/livestream",
                      help="RTMP stream URL (default)")
    input_group.add_argument("--video", type=str,
                      help="Local video file path for testing")
    
    parser.add_argument("--output", default="./yolo_output",
                      help="Output directory for results")
    # Removed --model argument to prevent downloading other models
    parser.add_argument("--conf", type=float, default=0.5,
                      help="Confidence threshold for detections")
    parser.add_argument("--iou", type=float, default=0.45,
                      help="IOU threshold for NMS")
    parser.add_argument("--no-segmentation", action="store_true",
                      help="Disable segmentation masks")
    
    args = parser.parse_args()
    
    # Determine input source and type
    if args.video:
        input_source = args.video
        is_video_file = True
        print(f"Using local video file: {input_source}")
    else:
        input_source = args.url
        is_video_file = False
        print(f"Using RTMP stream: {input_source}")
    
    # Create processor - always uses yolov8s.pt (small model)
    processor = YOLOv8StreamProcessor(
        input_source=input_source,
        output_dir=args.output,
        is_video_file=is_video_file
    )
    
    # Configure model parameters (model is fixed to yolov8s.pt)
    processor.confidence_threshold = args.conf
    processor.iou_threshold = args.iou
    processor.enable_segmentation = not args.no_segmentation
    
    print("Using YOLOv8 Small Model (yolov8s.pt) - no other models will be downloaded")
    
    # Start processing
    success = processor.start_processing()
    
    if not success:
        source_type = "video file" if is_video_file else "stream"
        print(f"Failed to start {source_type} processing")
        return 1
    
    source_type = "video file" if is_video_file else "stream"
    print(f"YOLOv8 {source_type} processing completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
