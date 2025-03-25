# Dockerfile
FROM python:3.10.16

# Set a working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . /app/

# Default command (just for demonstration)
CMD ["/bin/bash"]
