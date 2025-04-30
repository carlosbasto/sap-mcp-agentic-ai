FROM python:3.11-slim

WORKDIR /app

# Install uv for faster package management
RUN pip install uv

# Copy requirements file
COPY requirements.txt .

# Install dependencies using uv
RUN uv venv && uv pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .

# Expose the port the server runs on
EXPOSE 8050

# Command to run the server
CMD ["uv", "run", "server.py"] 