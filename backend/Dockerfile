# syntax=docker/dockerfile:1
FROM ubuntu:18.04

# Set the working directory in the container
WORKDIR ./app
# Copy the current directory contents into the container
COPY . .

# Install software-properties-common
RUN apt clean
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3 and Tesseract
RUN add-apt-repository "ppa:alex-p/tesseract-ocr"
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    apt-get install -y tesseract-ocr libtesseract-dev

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV Name GI-tool-backend

# Run application
CMD ["flask", "run"]
