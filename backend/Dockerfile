# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr libtesseract-dev libgl1 poppler-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 8000

# Use Gunicorn with threading to optimize for low CPU/memory
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "600", "--bind", "0.0.0.0:8000", "app:app"]
