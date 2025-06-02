```
ffmpeg \
  -f avfoundation -framerate 30 -video_size 1280x720 -i "0:0" \
  -vcodec libx264 -preset veryfast -g 50 -tune zerolatency \
  -acodec aac -ar 44100 -ac 2 -b:a 128k \
  -f flv rtmp://localhost/mystream
```