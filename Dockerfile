FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Working directory
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements_serverless.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements_serverless.txt

# Copy handler
COPY serverless_handler_fixed.py serverless_handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run handler
CMD ["python", "serverless_handler.py"]
