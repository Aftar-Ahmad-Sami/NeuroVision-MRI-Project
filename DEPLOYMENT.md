# Deployment Guide

This document provides instructions for deploying NeuroVision AI to various platforms.

## Hugging Face Spaces

### Hugging Face Spaces Configuration

Hugging Face Spaces requires:
- Port: 7860 (automatically set)
- Production server: gunicorn
- No debug mode

The Dockerfile is pre-configured for this.

### Deployment Steps

1. Create a new Space on Hugging Face
2. Choose "Docker" as the SDK
3. Push your code to the Space repository
4. Hugging Face will automatically build and deploy using the Dockerfile

### Environment Variables

The application automatically uses the correct port (7860) when deployed to Hugging Face Spaces. No manual configuration is needed.

## Local Development

For local development, you can run the Flask development server:

```bash
# Set environment variables (optional)
export PORT=5000
export FLASK_ENV=development

# Run the app
python app.py
```

Or use the production server locally:

```bash
gunicorn --bind 0.0.0.0:7860 --workers 1 --timeout 120 app:app
```

## Docker Deployment

Build and run the Docker container:

```bash
# Build the image
docker build -t neurovision-ai .

# Run the container
docker run -p 7860:7860 neurovision-ai
```

## Production Considerations

- Use gunicorn or another WSGI server (not Flask's development server)
- Disable debug mode in production
- Set appropriate timeout values for model loading (120 seconds recommended)
- Use single worker due to ML model memory requirements
- Ensure sufficient memory for PyTorch models
- Consider using GPU-enabled containers for better performance
