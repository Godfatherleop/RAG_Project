# Use a Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
COPY Knowledge_Base.pdf /app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Collect static files (if applicable)
RUN python manage.py collectstatic --noinput

# Expose the port
EXPOSE 8000

# Run the server (development: `runserver`, production: `gunicorn`)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
