FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY requirements.txt ./
COPY lin_reg.bin ./
RUN pip install -r requirements.txt

COPY starter.py ./

ENTRYPOINT ["python", "starter.py"]
