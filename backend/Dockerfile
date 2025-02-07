# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && \
    apt install -y libgl1 && \
    apt-get install -y poppler-utils

# Copy the current directory contents into the container
COPY . .

# Install any Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 8000

# Command to run the application
CMD ["gunicorn", "--timeout", "600","app:app"]