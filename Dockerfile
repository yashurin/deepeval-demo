# Image for the evaluation app. Ollama itself runs from the official
# ollama/ollama image declared in docker-compose.yml, not from here.
FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies before copying the app so this layer stays
# cached when only the scripts change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Print logs as they happen, skip DeepEval's anonymous telemetry, and
# refuse the post-run "inspect" prompt so the container can exit.
ENV PYTHONUNBUFFERED=1 \
    DEEPEVAL_TELEMETRY_OPT_OUT=1 \
    DEEPEVAL_NO_INSPECT_PROMPT=1 \
    DEEPEVAL_DISABLE_TIMEOUTS=1 \
    DEEPEVAL_DISABLE_DOTENV=1

# Default is a short run. Override the command for the full suite.
CMD ["python", "-m", "app.evaluate", "--quick"]
