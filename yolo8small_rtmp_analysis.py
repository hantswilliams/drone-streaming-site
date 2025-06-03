#!/usr/bin/env python3
"""
YOLO v8 RTMP Stream Analysis
Real-time object detection and segmentation using YOLOv8 on RTMP stream
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
    """Real-time RTMP stream processor with YOLOv8 object detection and segmentation"""

    def __init__(self, rtmp_url="rtmp://34.67.35.85:1935/live/livestream", output_dir="./yolo_output"):
        self.rtmp_url = rtmp_url
        self.output_dir = output_dir
        self.cap = None
        self.running = False
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
        
    def connect_to_stream(self):
        """Connect to RTMP stream"""
        print(f"Connecting to RTMP stream: {self.rtmp_url}")
        
        # Configure OpenCV for RTMP
        self.cap = cv2.VideoCapture(self.rtmp_url)
        
        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            print("Error: Could not connect to RTMP stream")
            return False
            
        # Get stream properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Stream properties: {width}x{height} @ {fps} FPS")
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
        while self.running:
            if self.cap is not None:
                ret, frame = self.cap.read()
                if ret:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                    else:
                        # Drop frame if queue is full (reduce latency)
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(frame)
                        except queue.Empty:
                            pass
                else:
                    print("Failed to read frame from stream")
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
    
    def start_processing(self):
        """Start the RTMP stream processing"""
        if not self.connect_to_stream():
            return False
        
        self.running = True
        self.start_time = time.time()
        
        # Start frame capture thread
        capture_thread = threading.Thread(target=self.frame_capture_worker)
        capture_thread.daemon = True
        capture_thread.start()
        
        print("Starting YOLOv8 stream processing...")
        print("Press 'q' to quit, 's' to save current frame")
        
        try:
            while self.running:
                try:
                    # Get frame from queue
                    frame = self.frame_queue.get(timeout=1.0)
                    self.frame_count += 1
                    
                    # Process frame with YOLOv8
                    annotated_frame, detections = self.process_frame_with_yolo(frame)
                    
                    if detections:
                        self.detection_count += len(detections)
                        print(f"Frame {self.frame_count}: Found {len(detections)} objects")
                        
                        # Save detection results
                        self.save_detection_results(detections, self.frame_count)
                        
                        # Save frame with detections
                        self.save_frame(annotated_frame, self.frame_count)
                    
                    # Calculate FPS
                    self.calculate_fps()
                    
                    # Display frame
                    cv2.imshow('YOLOv8 RTMP Stream Analysis', annotated_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quit requested by user")
                        break
                    elif key == ord('s'):
                        filename = self.save_frame(annotated_frame, self.frame_count, "manual_save")
                        print(f"Frame saved: {filename}")
                        
                except queue.Empty:
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
    parser = argparse.ArgumentParser(description="YOLOv8 Small Model RTMP Stream Analysis")
    parser.add_argument("--url", default="rtmp://34.67.35.85:1935/live/livestream",
                      help="RTMP stream URL")
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
    
    # Create processor - always uses yolov8s.pt (small model)
    processor = YOLOv8StreamProcessor(
        rtmp_url=args.url,
        output_dir=args.output
    )
    
    # Configure model parameters (model is fixed to yolov8s.pt)
    processor.confidence_threshold = args.conf
    processor.iou_threshold = args.iou
    processor.enable_segmentation = not args.no_segmentation
    
    print("Using YOLOv8 Small Model (yolov8s.pt) - no other models will be downloaded")
    
    # Start processing
    success = processor.start_processing()
    
    if not success:
        print("Failed to start stream processing")
        return 1
    
    print("YOLOv8 stream processing completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
