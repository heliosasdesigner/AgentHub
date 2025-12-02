FROM python:3.11-slim

WORKDIR /app

# Install uv package manager
RUN pip install --no-cache-dir uv

# Pre-copy dependency files to leverage Docker cache
COPY requirements.txt pyproject.toml /app/

# Install dependencies system-wide using uv
RUN uv pip install --system -r /app/requirements.txt

COPY backend /app/backend

WORKDIR /app/backend

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "11111"]
