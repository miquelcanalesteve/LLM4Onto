# Dockerfile.model
FROM python:3.10

WORKDIR /app

# Copy dependencies and install them
COPY requirements/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Keep the container open for manual script execution
CMD ["/bin/bash"]
