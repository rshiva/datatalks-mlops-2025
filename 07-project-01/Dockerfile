
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local pipelines directory to the container
COPY pipelines/ ./pipelines

# Set the command to run when the container starts
# This will run the batch prediction script by default
CMD ["python", "pipelines/batch_predict.py"]
