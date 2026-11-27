FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY pipeline.py .
COPY output/model/svm_final.joblib output/model/svm_final.joblib
COPY output/rekomendasi_pengendalian.json output/rekomendasi_pengendalian.json

CMD ["python", "-u", "pipeline.py"]
