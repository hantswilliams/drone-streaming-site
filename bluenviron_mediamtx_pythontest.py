#!/usr/bin/env python3
"""
RTMP Stream Capture and Computer Vision Analysis
Captures RTMP stream and performs real-time computer vision processing
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

class RTMPStreamProcessor:
    """Real-time RTMP stream processor with computer vision capabilities"""
    
    def __init__(self, rtmp_url="rtmp://localhost/mystream", output_dir="./cv_output"):
        self.rtmp_url = rtmp_url
        self.output_dir = output_dir
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.detection_results = []
        
        # Computer Vision Parameters
        self.enable_motion_detection = True
        self.enable_object_detection = False
        self.enable_face_detection = True
        self.enable_edge_detection = True
        
        # Motion detection parameters
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.motion_threshold = 1000  # Minimum area for motion detection
        
        # Face detection
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            print("Warning: Face detection cascade not found. Face detection disabled.")
            self.enable_face_detection = False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.fps = 0
    
    def connect_to_stream(self):
        """Connect to RTMP stream"""
        print(f"Connecting to RTMP stream: {self.rtmp_url}")
        
        # Try different connection methods
        connection_attempts = [
            self.rtmp_url,
            f"{self.rtmp_url}?timeout=10000000",
            self.rtmp_url.replace("rtmp://", "rtmp://")
        ]
        
        for attempt_url in connection_attempts:
            try:
                self.cap = cv2.VideoCapture(attempt_url)
                
                # Set buffer size to reduce latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Check if connection is successful
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"Successfully connected to stream: {attempt_url}")
                        print(f"Frame size: {frame.shape}")
                        return True
                    else:
                        print(f"Failed to read frame from: {attempt_url}")
                        self.cap.release()
                else:
                    print(f"Failed to open: {attempt_url}")
                    
            except Exception as e:
                print(f"Error connecting to {attempt_url}: {e}")
                if self.cap:
                    self.cap.release()
        
        print("Failed to connect to RTMP stream")
        return False
    
    def detect_motion(self, frame):
        """Detect motion in frame"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        motion_areas = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.motion_threshold:
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append((x, y, w, h, area))
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Motion: {int(area)}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame, motion_detected, motion_areas, fg_mask
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        if not self.enable_face_detection:
            return frame, []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_data = []
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            face_data.append((x, y, w, h))
        
        return frame, face_data
    
    def detect_edges(self, frame):
        """Detect edges using Canny edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Convert back to BGR for display
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return edges_bgr
    
    def analyze_frame(self, frame):
        """Perform comprehensive frame analysis"""
        original_frame = frame.copy()
        results = {
            'timestamp': datetime.now().isoformat(),
            'frame_number': self.frame_count,
            'motion_detected': False,
            'faces_detected': 0,
            'motion_areas': [],
            'face_locations': []
        }
        
        # Motion detection
        if self.enable_motion_detection:
            frame, motion_detected, motion_areas, motion_mask = self.detect_motion(frame)
            results['motion_detected'] = motion_detected
            results['motion_areas'] = motion_areas
        
        # Face detection
        if self.enable_face_detection:
            frame, faces = self.detect_faces(frame)
            results['faces_detected'] = len(faces)
            results['face_locations'] = faces
        
        # Edge detection (for overlay)
        if self.enable_edge_detection:
            edges = self.detect_edges(original_frame)
        
        # Add timestamp and info overlay
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save detection results
        self.detection_results.append(results)
        
        return frame, results
    
    def capture_frames(self):
        """Capture frames from RTMP stream"""
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                print("Failed to read frame from stream")
                time.sleep(0.1)
                continue
            
            # Add frame to queue (non-blocking)
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                # Remove oldest frame if queue is full
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame, block=False)
                except queue.Empty:
                    pass
    
    def process_frames(self):
        """Process frames with computer vision"""
        cv2.namedWindow('RTMP Stream Analysis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RTMP Stream Analysis', 1200, 800)
        
        # Create windows for different views
        if self.enable_motion_detection:
            cv2.namedWindow('Motion Mask', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Motion Mask', 400, 300)
        
        if self.enable_edge_detection:
            cv2.namedWindow('Edge Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Edge Detection', 400, 300)
        
        while self.running:
            try:
                # Get latest frame
                frame = self.frame_queue.get(timeout=1.0)
                
                # Process frame
                processed_frame, results = self.analyze_frame(frame)
                
                # Update statistics
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.start_time)
                    self.last_fps_time = current_time
                
                # Display processed frame
                cv2.imshow('RTMP Stream Analysis', processed_frame)
                
                # Show additional views
                if self.enable_motion_detection and 'motion_detected' in results:
                    # Show motion mask
                    _, _, _, motion_mask = self.detect_motion(frame.copy())
                    cv2.imshow('Motion Mask', motion_mask)
                
                if self.enable_edge_detection:
                    edges = self.detect_edges(frame)
                    cv2.imshow('Edge Detection', edges)
                
                # Save interesting frames
                if results['motion_detected'] or results['faces_detected'] > 0:
                    self.save_detection_frame(processed_frame, results)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit requested by user")
                    self.running = False
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"{self.output_dir}/manual_save_{self.frame_count}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Frame saved: {filename}")
                elif key == ord('m'):
                    # Toggle motion detection
                    self.enable_motion_detection = not self.enable_motion_detection
                    print(f"Motion detection: {'ON' if self.enable_motion_detection else 'OFF'}")
                elif key == ord('f'):
                    # Toggle face detection
                    self.enable_face_detection = not self.enable_face_detection
                    print(f"Face detection: {'ON' if self.enable_face_detection else 'OFF'}")
                elif key == ord('e'):
                    # Toggle edge detection
                    self.enable_edge_detection = not self.enable_edge_detection
                    print(f"Edge detection: {'ON' if self.enable_edge_detection else 'OFF'}")
                
            except queue.Empty:
                print("No frames available, waiting...")
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
    
    def save_detection_frame(self, frame, results):
        """Save frame when detection occurs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if results['motion_detected']:
            filename = f"{self.output_dir}/motion_{timestamp}_{self.frame_count}.jpg"
            cv2.imwrite(filename, frame)
        
        if results['faces_detected'] > 0:
            filename = f"{self.output_dir}/faces_{timestamp}_{self.frame_count}.jpg"
            cv2.imwrite(filename, frame)
    
    def save_analysis_results(self):
        """Save detection results to JSON file"""
        results_file = f"{self.output_dir}/analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            'total_frames': self.frame_count,
            'duration_seconds': time.time() - self.start_time,
            'average_fps': self.fps,
            'motion_detections': sum(1 for r in self.detection_results if r['motion_detected']),
            'face_detections': sum(1 for r in self.detection_results if r['faces_detected'] > 0),
            'detections': self.detection_results[-100:]  # Save last 100 detections
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Analysis results saved to: {results_file}")
    
    def start_processing(self):
        """Start the stream processing"""
        if not self.connect_to_stream():
            return False
        
        self.running = True
        self.start_time = time.time()
        
        # Start capture thread
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.daemon = True
        capture_thread.start()
        
        # Start processing in main thread
        try:
            print("Starting stream processing...")
            print("Controls:")
            print("  'q' - Quit")
            print("  's' - Save current frame")
            print("  'm' - Toggle motion detection")
            print("  'f' - Toggle face detection")
            print("  'e' - Toggle edge detection")
            print()
            
            self.process_frames()
            
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop_processing()
        
        return True
    
    def stop_processing(self):
        """Stop the stream processing"""
        print("Stopping stream processing...")
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Save analysis results
        if self.detection_results:
            self.save_analysis_results()
        
        # Print summary
        duration = time.time() - self.start_time
        print(f"\nProcessing Summary:")
        print(f"  Total frames processed: {self.frame_count}")
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  Average FPS: {self.fps:.1f}")
        print(f"  Motion detections: {sum(1 for r in self.detection_results if r['motion_detected'])}")
        print(f"  Face detections: {sum(1 for r in self.detection_results if r['faces_detected'] > 0)}")

def main():
    parser = argparse.ArgumentParser(description='RTMP Stream Computer Vision Processor')
    parser.add_argument('--url', default='rtmp://localhost/mystream',
                       help='RTMP stream URL')
    parser.add_argument('--output', default='./cv_output',
                       help='Output directory for saved frames and results')
    parser.add_argument('--no-motion', action='store_true',
                       help='Disable motion detection')
    parser.add_argument('--no-faces', action='store_true',
                       help='Disable face detection')
    parser.add_argument('--no-edges', action='store_true',
                       help='Disable edge detection')
    
    args = parser.parse_args()
    
    print("RTMP Stream Computer Vision Processor")
    print("=" * 50)
    print(f"Stream URL: {args.url}")
    print(f"Output directory: {args.output}")
    print()
    
    # Create processor
    processor = RTMPStreamProcessor(
        rtmp_url=args.url,
        output_dir=args.output
    )
    
    # Configure features
    if args.no_motion:
        processor.enable_motion_detection = False
    if args.no_faces:
        processor.enable_face_detection = False
    if args.no_edges:
        processor.enable_edge_detection = False
    
    # Start processing
    success = processor.start_processing()
    
    if not success:
        print("Failed to start stream processing")
        return 1
    
    print("Stream processing completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())