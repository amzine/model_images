# Use an official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code and models into the container
COPY . .

# Expose the port your app runs on (e.g., Flask defaults to 5000)
EXPOSE 5002

# Command to run the app
CMD ["python", "app.py"]
