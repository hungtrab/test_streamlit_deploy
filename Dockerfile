# index_stock_predictor_mlops/Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tạo các thư mục cần thiết bên trong image TRƯỚC KHI COPY hoặc MOUNT
RUN mkdir -p /app/application
RUN mkdir -p /app/data_volume

# Nơi data CSV sẽ được mount (config.CONTAINER_DATA_DIRECTORY)
# Thư mục models_store và database_files sẽ nằm trong /app/application (do COPY ./app)
# và được quản lý bởi volume mounts trong docker-compose

COPY ./app /app/application

EXPOSE 8000

# CMD mặc định cho API server
CMD ["python", "-m", "uvicorn", "application.api_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
