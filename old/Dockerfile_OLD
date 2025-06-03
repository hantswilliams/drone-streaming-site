# Dockerfile for MediaMTX streaming server
FROM bluenviron/mediamtx:1.8.5

# Set environment variables for RTSP and WebRTC configuration
ENV MTX_RTSPTRANSPORTS=tcp
ENV MTX_WEBRTCADDITIONALHOSTS=20.62.193.224

# Expose the required ports
# RTMP port  
EXPOSE 1935
# HTTP/WebRTC ports
EXPOSE 8888


# Set the working directory
WORKDIR /


