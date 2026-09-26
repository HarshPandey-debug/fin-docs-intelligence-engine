FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN addgroup --system app && adduser --system --ingroup app app
COPY frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY frontend/app.py ./app.py
USER app

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

