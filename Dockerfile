# Dockerfile for SRS (Simple Realtime Server)
FROM ossrs/srs:5

# Expose the required ports
# RTMP port
EXPOSE 1935
# HTTP API port
EXPOSE 1985
# HTTP server port
EXPOSE 8080
# WebRTC UDP ports
EXPOSE 8000/udp
EXPOSE 10080/udp

# Set the working directory
WORKDIR /usr/local/srs

