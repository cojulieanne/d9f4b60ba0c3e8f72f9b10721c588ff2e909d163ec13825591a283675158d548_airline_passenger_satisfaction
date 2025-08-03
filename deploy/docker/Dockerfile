# Base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install project dependencies
COPY pyproject.toml ./
RUN pip install uv && uv pip install -r pyproject.toml

# Copy the source code
COPY src/ ./src

# Create folders for input/output if needed
RUN mkdir -p data results

# Run the ML pipeline when the container starts
CMD ["python", "src/pipeline.py"]
