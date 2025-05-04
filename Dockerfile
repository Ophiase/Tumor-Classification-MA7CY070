# Use the NVIDIA CUDA base image with Python 3.12
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install Python and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip && \
    ln -s /usr/bin/python3.12 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the environment files to the container
COPY requirements.txt requirements-cuda.txt environment.yml ./

# Install Python dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the default command (optional, uncomment if needed)
# CMD ["python", "your_script.py"]
