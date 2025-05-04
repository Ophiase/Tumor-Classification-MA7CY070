# Use the official Python 3.12.7 image as the base image
FROM python:3.12.7-slim
# Set the working directory in the container
WORKDIR /app

#  --------------------------------------------------
    
# Copy the environment files to the container
COPY requirements.txt environment.yml ./
# If using pip and requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# If using conda and environment.yml
# RUN apt-get update && apt-get install -y wget && \
#     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
#     bash miniconda.sh -b -p /opt/conda && \
#     rm miniconda.sh && \
#     /opt/conda/bin/conda env create -f environment.yml && \
#     echo "source activate $(head -n 1 environment.yml | cut -d ' ' -f2)" > ~/.bashrc
# Copy the rest of the application code
COPY . .

#  --------------------------------------------------

# Expose the port for Jupyter Notebook
# EXPOSE 8888
# Default command to run Jupyter Notebook
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]