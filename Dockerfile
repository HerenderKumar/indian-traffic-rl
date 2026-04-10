# 1. Use a standard, stable Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install necessary system build tools (prevents C++ compiler errors)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your requirements file into the container
COPY requirements.txt .

# 5. PLAN B: The Magic Step
# We ignore the complex uv/lockfiles and force standard pip to install 
# exactly what we need globally inside the container.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir openai numpy torch fastapi uvicorn pydantic pyyaml openenv-core requests httpx

# 6. Copy the rest of your project files (inference.py, maps, models)
COPY . .

# 7. Expose the port your FastAPI server runs on
EXPOSE 8000

# 8. Start the FastAPI server (Adjust if your main server file is named differently)
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]