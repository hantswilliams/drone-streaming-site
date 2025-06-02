# Dockerfile for MediaMTX streaming server
FROM bluenviron/mediamtx:latest

# Set environment variables for RTSP and WebRTC configuration
ENV MTX_RTSPTRANSPORTS=tcp
ENV MTX_WEBRTCADDITIONALHOSTS=192.168.x.x

# Expose the required ports
# RTSP port
EXPOSE 8554
# RTMP port  
EXPOSE 1935
# HTTP/WebRTC ports
EXPOSE 8888
EXPOSE 8889
# UDP ports for WebRTC
EXPOSE 8890/udp
EXPOSE 8189/udp

# Set the working directory
WORKDIR /

# The base image already has the entrypoint configured
# No additional commands needed as MediaMTX will start automatically

# Optional: Add labels for documentation
LABEL maintainer="drone-streaming-site"
LABEL description="MediaMTX streaming server for drone video streaming"
LABEL version="1.0"

# Optional: Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/ || exit 1
