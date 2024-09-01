# Use a base image of Python with version 3.9
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file (if it exists) into the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# Command to run your Python script
CMD ["python", "main.py"]

# 1.docker build -t imdb_predictor .
# 2.docker run imdb_predictor   
