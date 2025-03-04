# Use the official TensorFlow base image (includes TensorFlow and Keras)
FROM tensorflow/tensorflow:latest

# Set the working directory in the container
WORKDIR /app

# Copy your application code into the container
COPY . .

# Upgrade pip and install dependencies directly
RUN pip install --upgrade pip && \
    pip install --ignore-installed numpy pandas hyperopt scikit-learn mlflow

# Set the command to run your training script
CMD ["python", "train.py"]
