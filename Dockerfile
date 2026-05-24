FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose ports
# 7860: Gradio UI
# 8000: API Deployment Server
EXPOSE 7860
EXPOSE 8000

# Ensure Gradio listens on all interfaces
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Start the application
CMD ["python", "app.py"]
