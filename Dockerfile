# 1. Use a standard, stable Python base image
FROM python:3.10-slim-bookworm

# Prevent Python from buffering stdout/stderr (vital for logs)
ENV PYTHONUNBUFFERED=1

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install necessary system build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your requirements file into the container
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 6. Install CPU-only PyTorch (keeps image small)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 7. Copy the rest of your project files
COPY . .

# 8. Expose the port defined in openenv.yaml
EXPOSE 7860

# 9. Run the FastAPI server from server/app.py
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]