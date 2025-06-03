

## Docker run command: 


# Run the container
```bash
docker run --rm -it \
  -p 1935:1935 \
  -p 8888:8888 \
  drone
```


```bash
docker run --rm -it \
-e MTX_RTSPTRANSPORTS=tcp \
-e MTX_WEBRTCADDITIONALHOSTS=192.168.x.x \
-p 8554:8554 \
-p 1935:1935 \
-p 8888:8888 \
-p 8889:8889 \
-p 8890:8890/udp \
-p 8189:8189/udp \
bluenviron/mediamtx
```

### Working RTMP
```bash
ffmpeg \
  -f avfoundation -framerate 30 -video_size 1280x720 -i "0:0" \
  -vcodec libx264 -preset veryfast -g 50 -tune zerolatency \
  -acodec aac -ar 44100 -ac 2 -b:a 128k \
  -f flv rtmp://localhost/mystream
```


### Working RTSP 
```bash
ffmpeg \
  -f avfoundation -framerate 30 -video_size 1280x720 -i "0:0" \
  -vcodec libx264 -preset veryfast -g 50 -tune zerolatency \
  -acodec aac -ar 44100 -ac 2 -b:a 128k \
  -f rtsp rtsp://localhost:8554/mystream
```


## Capture to OpenCV
```python
myrtmp_addr = "rtp://localhost:8554/mystream"
cap = cv2.VideoCapture(myrtmp_addr)
frame,err = cap.read()
```