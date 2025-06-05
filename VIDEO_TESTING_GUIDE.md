# YOLOv8 Video File Testing Guide

This guide explains how to use the YOLOv8StreamProcessor with local video files for testing different model capabilities.

## Overview

The `yolo8small_rtmp_analysis.py` script now supports both RTMP streams and local video files as input sources. This allows you to:

- Test object detection on recorded videos
- Evaluate model performance on different types of content
- Debug and fine-tune detection parameters offline
- Process video files without needing a live stream

## Usage

### Command Line Interface

#### Process a local video file:
```bash
python yolo8small_rtmp_analysis.py --video /path/to/your/video.mp4
```

#### Process an RTMP stream (default behavior):

- For starting a stream to a remote RTMP server already in place:
```bash
ffmpeg \
  -f avfoundation -framerate 30 -video_size 1280x720 -i "0:0" \
  -vcodec libx264 -preset veryfast -g 50 -tune zerolatency \
  -acodec aac -ar 44100 -ac 2 -b:a 128k \
  -f flv rtmp://34.67.35.85/live/livestream
```

- For analyzing the stream: 
```bash
python yolo8small_rtmp_analysis.py --url rtmp://your-stream-url
```

#### Advanced options:
```bash
# Custom output directory and detection parameters
python yolo8small_rtmp_analysis.py --video ./test_videos/sample.mp4 \
    --output ./custom_output \
    --conf 0.3 \
    --iou 0.5 \
    --no-segmentation
```

### Using the Test Script

The `test_local_video.py` script provides a simple way to test video file processing:

```bash
# Basic usage
python test_local_video.py ./test_videos/sample_video.mp4

# With custom output directory
python test_local_video.py ./test_videos/sample_video.mp4 ./my_results
```

## Supported Video Formats

The system supports all video formats that OpenCV can read:
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WebM (.webm)
- And many others

## Key Differences: Stream vs Video File

| Feature | RTMP Stream | Video File |
|---------|-------------|------------|
| **Input Source** | Live stream URL | Local file path |
| **Processing** | Real-time | Can be faster than real-time |
| **Ending** | Manual quit (press 'q') | Automatic when video ends |
| **Buffering** | Minimal buffer for low latency | Normal file reading |
| **Duration Info** | Not available | Shows duration and frame count |

## Example Usage Scenarios

### 1. Testing Detection Accuracy
```bash
# Use a lower confidence threshold to see more detections
python yolo8small_rtmp_analysis.py --video test_video.mp4 --conf 0.2
```

### 2. Performance Benchmarking
```bash
# Process without segmentation for faster performance
python yolo8small_rtmp_analysis.py --video large_video.mp4 --no-segmentation
```

### 3. Batch Processing Setup
```bash
# Process multiple videos with different parameters
for video in ./test_videos/*.mp4; do
    python yolo8small_rtmp_analysis.py --video "$video" --output "./results/$(basename "$video" .mp4)"
done
```

## Output Structure

When processing video files, the output structure is the same as for streams:

```
output_directory/
├── detections_YYYYMMDD_HHMMSS_frame_N.json    # Detection results per frame
├── detection_YYYYMMDD_HHMMSS_N.jpg            # Annotated frames with detections
├── manual_save_YYYYMMDD_HHMMSS_N.jpg          # Manually saved frames (press 's')
└── analysis_stats_YYYYMMDD_HHMMSS.json        # Final processing statistics
```

## Interactive Controls

During video processing, you can use these keyboard controls:

- **'q'**: Quit processing early
- **'s'**: Save the current frame as a manual save
- **ESC**: Alternative quit method

## Tips for Testing

1. **Start with short videos** (30-60 seconds) to quickly test functionality
2. **Use videos with known objects** to verify detection accuracy
3. **Try different confidence thresholds** to find optimal settings
4. **Compare results** between segmentation enabled/disabled
5. **Monitor processing speed** - video files often process faster than real-time

## Model Information

- **Model Used**: YOLOv8 Small (yolov8s.pt)
- **Model Size**: ~22MB
- **Detection Classes**: 80 COCO classes (person, car, dog, etc.)
- **Capabilities**: Object detection + instance segmentation

## Troubleshooting

### Video file not found
```
Error: Video file not found: /path/to/video.mp4
```
- Check the file path is correct
- Ensure the file exists and is readable

### Codec issues
```
Error: Could not open video file
```
- Try converting the video to MP4 format
- Ensure OpenCV is compiled with the necessary codecs

### Performance issues
- Use `--no-segmentation` for faster processing
- Increase confidence threshold to reduce detections
- Try smaller video files first

## Example Test Videos

You can download test videos from:
- [Sample Videos Repository](https://sample-videos.com/)
- [Pixabay Free Videos](https://pixabay.com/videos/)
- [Pexels Free Videos](https://www.pexels.com/videos/)

Place test videos in the `./test_videos/` directory for easy access.
