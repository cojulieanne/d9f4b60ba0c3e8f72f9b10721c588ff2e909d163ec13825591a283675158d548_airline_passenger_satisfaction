# Base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y libgomp1 && apt-get clean

# Copy and install project dependencies
COPY pyproject.toml ./
RUN pip install --root-user-action=ignore uv && uv pip install --system -r pyproject.toml

# Copy the source code
COPY src/ ./src

# Create folders for input/output if needed
RUN mkdir -p data results

# Run the ML pipeline when the container starts
CMD ["python", "src/pipeline.py"]
