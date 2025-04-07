# Step 1: Build the Next.js frontend
FROM node:18 AS build-frontend

# Set working directory for the frontend
WORKDIR /app/frontend

# Install dependencies and build the Next.js app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install
COPY frontend ./
RUN npm run build

# Step 2: Set up the Python backend
FROM python:3.9-slim AS build-backend

# Set working directory for the backend
WORKDIR /app/backend

# Install system dependencies for PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with specific versions
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Copy all backend files
COPY backend/ ./

# Step 3: Final image setup
FROM python:3.9-slim

# Install system dependencies for PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for serving the frontend
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
RUN apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy the frontend build and backend code
COPY --from=build-frontend /app/frontend/.next ./frontend/.next
COPY --from=build-frontend /app/frontend/public ./frontend/public
COPY --from=build-frontend /app/frontend/package.json ./frontend/package.json

# Create a basic next.config.js if it doesn't exist
RUN echo 'module.exports = { output: "standalone" }' > ./frontend/next.config.js

# Copy all backend files and Python packages from build-backend stage
COPY --from=build-backend /app/backend ./backend
COPY --from=build-backend /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Install frontend dependencies
WORKDIR /app/frontend
RUN npm install --production

# Expose ports
EXPOSE 3000 5000

# Create a script to run both services
WORKDIR /app
RUN echo '#!/bin/bash\n\
cd /app/frontend && npm start & \
cd /app/backend && python app.py' > /app/start.sh && \
chmod +x /app/start.sh

# Command to run both services
CMD ["/app/start.sh"]
