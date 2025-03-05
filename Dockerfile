# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the script into the container
COPY train.py .

# Install MLflow
RUN pip install --no-cache-dir mlflow

# Run the script when the container starts
CMD ["python", "train.py"]