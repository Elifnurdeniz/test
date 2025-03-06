# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the script into the container
COPY train.py .

# Install Python dependencies
RUN pip install --no-cache-dir mlflow pandas scikit-learn \
    numpy \
    google-auth \
    google-auth-oauthlib \
    google-auth-httplib2 \
    google-cloud-storage \
    gcloud \
    pymysql \
    cryptography \
    google

# Run the script when the container starts
CMD ["python", "train.py"]