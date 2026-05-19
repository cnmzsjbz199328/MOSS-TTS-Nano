FROM continuumio/miniconda3:latest

WORKDIR /code

# Install system dependencies for audio and compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create target directories
RUN mkdir -p /code/generated_audio /code/models

# Install pynini via conda-forge to bypass OpenFST compilation issues on Linux
RUN conda install -y -c conda-forge pynini=2.1.6.post1 python=3.12

# Copy requirements
COPY requirements.txt /code/requirements.txt

# Install PyTorch CPU to keep the Docker image small and download fast
RUN pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy all repository files
COPY . /code

# Set permission so the Hugging Face Space runtime (which runs as user 1000) can read/write models
RUN chmod -R 777 /code

# Set user to 1000 as per HF Space guidelines
USER 1000

# Set Hugging Face cache and environment directories
ENV HOME=/code \
    PYTHONUNBUFFERED=1

# Expose port 7860
EXPOSE 7860

# Start the web demo with the ONNX backend for CPU-friendly inference
CMD ["python", "app_onnx.py", "--host", "0.0.0.0", "--port", "7860"]
