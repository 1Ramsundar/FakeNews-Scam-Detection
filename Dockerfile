FROM python:3.10-slim

# System deps (Tesseract for OCR)
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr git && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install Python deps first (cache layer)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user . .

EXPOSE 7860

CMD ["python", "app.py"]
