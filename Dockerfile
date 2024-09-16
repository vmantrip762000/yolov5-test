# Base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies with PyTorch CPU version
RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2 -f https://download.pytorch.org/whl/torch_stable.html

# Install remaining dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code into the container
COPY src/ .

# Command to run application
CMD ["python", "app.py"]
