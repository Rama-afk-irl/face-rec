# Dockerfile

# Use a standard, slim version of Python as our base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Update package lists and install the system dependencies we need
RUN apt-get update && apt-get install -y cmake build-essential

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install the Python packages
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your application code into the container
COPY . /code/
