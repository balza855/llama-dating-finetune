FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Working directory
WORKDIR /app

# Install git (git hatası için)
RUN apt-get update && apt-get install -y git

# Copy requirements
COPY requirements_serverless.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements_serverless.txt

# Copy handler
COPY serverless_handler.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run handler
CMD ["python", "serverless_handler.py"]
