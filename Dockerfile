FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN rm -rf venv

# Mandatory environment variables per submission spec
# Override these via HuggingFace Space Secrets at runtime
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""
ENV ENV_BASE_URL="http://localhost:7860"

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start server via python directly as required
CMD ["python", "server/app.py"]