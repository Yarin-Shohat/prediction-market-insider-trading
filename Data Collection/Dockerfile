# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir requests beautifulsoup4

# Set default command to run the script
CMD ["python", "main.py"]
