# Simple start

## Instructions for VM 
- Need to make sure TCP 1935 and 8080 are open 
- Need to make sure UDP 8000 and 10080 are open 
- Install docker, then either run dockerfile or copy commands below: 

```bash
docker run --rm -it -p 1935:1935 -p 1985:1985 -p 8080:8080 \
    -p 8000:8000/udp -p 10080:10080/udp ossrs/srs:5
```

- Take note of the VMs IP address, can then start to stream a feed to: 
    - `rtmp://{IP_ADDRESS_HERE}/live/livestream` 
    - As a example, for starting to stream your cam to the server: 
```bash
ffmpeg \
  -f avfoundation -framerate 30 -video_size 1280x720 -i "0:0" \
  -vcodec libx264 -preset veryfast -g 50 -tune zerolatency \
  -acodec aac -ar 44100 -ac 2 -b:a 128k \
  -f flv rtmp://34.67.35.85/live/livestream
```
- Can then open up VLC player to `rtmp://34.67.35.85/live/livestream` , or the `opencv_rtmp_analysis.py` for openCV image analysis 

