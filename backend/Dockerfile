# syntax=docker/dockerfile:1
FROM ubuntu:latest

# Set the working directory in the container
WORKDIR ./app
# Copy the current directory contents into the container
COPY . .

# Install software-properties-common
# RUN apt clean
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y software-properties-common

# Install CV2 dependencies
RUN add-apt-repository universe && \
    apt upgrade -y
RUN apt-get install -y ffmpeg libsm6 libxext6

# Install Python 3 and Tesseract

# RUN add-apt-repository "ppa:alex-p/tesseract-ocr" && \
#    apt-get update

RUN apt-get install -y python3-pip python3-dev && \
    apt-get install -y tesseract-ocr libtesseract-dev

# Install tesseract languages
RUN apt-get install -y tesseract-ocr-rus

# Purging apt-get state information
RUN rm -rf /var/lib/apt/lists/*

# Download EAST model
# RUN wget -O "./app/data/frozen_east_text_detection.tar.gz" "https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1"
# RUN tar -xvf "./app/data/frozen_east_text_detection.tar.gz"

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Port
EXPOSE 5000

# Define environment variables
ENV Name GI-tool-backend

# Run application
CMD ["flask", "run"]
