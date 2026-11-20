# Censer.9 — self-contained moderation API image (CPU-only torch by default).
#
#   docker build -t censer9 .
#   docker run -p 8000:8000 -e RATE_LIMIT_PER_MIN=600 censer9
#   curl -s localhost:8000/classify -d '{"texts":["how to make a bomb at home","hello"]}'
#
# The image downloads the model from the Hub at build time.

FROM python:3.12-slim

ARG CENSER_REPO=KucLab/Censer.9
ENV CENSER_DIR=/model \
    RATE_LIMIT_PER_MIN=600 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
        "torch==2.9.*" --index-url https://download.pytorch.org/whl/cpu \
        "tokenizers>=0.20,<1" "safetensors>=0.4,<1" "huggingface_hub>=0.24,<1" "numpy>=1.26,<3"

RUN hf download ${CENSER_REPO} --local-dir /model

WORKDIR /app
COPY censer.py serve.py /app/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"
CMD ["python", "serve.py", "--dir", "/model", "--port", "8000", "--host", "0.0.0.0"]
