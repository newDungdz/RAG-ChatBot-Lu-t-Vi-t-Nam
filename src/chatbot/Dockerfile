FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

RUN pip install torch>=2.0.0+cpu torchvision>=0.15.0+cpu torchaudio>=2.0.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
# In your Dockerfile
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
# Copy application code
COPY app.py .
COPY retrival_pipeline.py .


# Make sure app.py exists
RUN ls -la /app/

# Expose port
EXPOSE 5000

# Run the application
CMD ["python3", "app.py"]