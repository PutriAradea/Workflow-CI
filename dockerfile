FROM python:3.12-slim

WORKDIR /app

COPY MLProject ./MLProject

RUN pip install --no-cache-dir pandas scikit-learn mlflow matplotlib

CMD ["python", "MLProject/modelling.py"]