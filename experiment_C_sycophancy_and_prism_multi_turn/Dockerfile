# ==============================================================================
# CS612 AI Systems & Evaluation - Unified Experiment Reproduction Container
# Evaluates Political Neutrality, Robustness, Sycophancy, and Locality Biases
# ==============================================================================

FROM python:3.10-slim

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Install Python requirements first (leverage Docker build cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy complete repository assets into container
COPY . .

# Ensure all execution scripts are executable
RUN chmod +x run_all.sh verify_repo.py

# Default execution: run the complete offline reproducible mock pipeline
# To run full evaluation against Ollama/vLLM, pass 'full' argument:
# docker run --rm cs612-eval-suite full
ENTRYPOINT ["./run_all.sh"]
CMD ["mock"]
