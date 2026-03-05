# Use Python 3.11 slim image (smaller than full Python)
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (needed for pdfplumber, numpy, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the AI model during build (so it doesn't download at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy all application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 5002

# Command to run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5002"]
