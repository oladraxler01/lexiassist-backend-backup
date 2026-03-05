# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (lighter - no AI model needed here)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies (no sentence-transformers needed here)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Expose port
EXPOSE 8083

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8083"]
