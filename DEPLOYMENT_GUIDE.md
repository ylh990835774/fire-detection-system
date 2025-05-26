# 🚀 火点检测系统部署指南

本指南将帮助您在不同环境中部署火点检测系统。

## 📋 目录

- [环境要求](#环境要求)
- [模型训练部署](#模型训练部署)
- [推理服务部署](#推理服务部署)
- [Docker 部署](#docker部署)
- [云平台部署](#云平台部署)
- [硬件优化配置](#硬件优化配置)
- [监控与维护](#监控与维护)

## 🔧 环境要求

### 基础环境

- Python 3.8+
- CUDA 11.0+ (NVIDIA GPU)
- ROCm 5.0+ (AMD GPU)
- 算能 SDK (算能 AI 芯片)

### 推荐硬件配置

#### 训练环境

- **GPU**: NVIDIA RTX 3080/4080 或更高
- **内存**: 32GB+ RAM
- **存储**: 500GB+ SSD
- **CPU**: Intel i7/AMD Ryzen 7 或更高

#### 推理环境

- **GPU**: NVIDIA GTX 1660 或更高
- **内存**: 16GB+ RAM
- **存储**: 100GB+ SSD
- **CPU**: Intel i5/AMD Ryzen 5 或更高

## 🏋️ 模型训练部署

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv fire_detection_env
source fire_detection_env/bin/activate  # Linux/Mac
# fire_detection_env\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 创建数据目录结构
mkdir -p data/fire_dataset/{images,labels}

# 数据格式验证
python -c "
from training.data_manager import DataManager
dm = DataManager('./data')
result = dm.validate_dataset('./data/fire_dataset/dataset.yaml')
print('数据集验证结果:', result['valid'])
"
```

### 3. 训练配置

```yaml
# configs/training_config.yaml
model_name: yolov8n
input_size: 640
epochs: 100
batch_size: 16
learning_rate: 0.01
device: auto
save_dir: runs/train
workers: 8

# 数据增强
augmentation: true
mixup: 0.0
copy_paste: 0.0

# 硬件优化
amp: true # 混合精度训练
sync_bn: true # 同步BatchNorm
```

### 4. 启动训练

```bash
# 方式1: 命令行训练
python training/train.py \
    --images-dir ./data/fire_dataset/images \
    --labels-dir ./data/fire_dataset/labels \
    --epochs 100 \
    --batch-size 16 \
    --model yolov8n \
    --device auto

# 方式2: 配置文件训练
python training/train.py \
    --config configs/training_config.yaml \
    --data-config ./data/fire_dataset/dataset.yaml

# 方式3: 分布式训练 (多GPU)
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    training/train.py \
    --device 0,1 \
    --batch-size 32
```

### 5. 模型验证与导出

```bash
# 验证模型
python training/train.py \
    --validate \
    --model-path runs/train/weights/best.pt \
    --data-config ./data/fire_dataset/dataset.yaml

# 导出ONNX模型
python -c "
from training.model_trainer import ModelTrainer
trainer = ModelTrainer()
trainer.export_model(
    model_path='runs/train/weights/best.pt',
    export_formats=['onnx', 'engine']
)
"
```

## 🌐 推理服务部署

### 1. 快速本地部署

```bash
# 启动Web API服务
python inference/deploy.py \
    --model-path ./models/fire_detection.onnx \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4

# 测试API
curl -X POST "http://localhost:8000/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.jpg"
```

### 2. 配置文件部署

```yaml
# configs/deploy_config.yaml
host: 0.0.0.0
port: 8000
workers: 4
api_title: "火点检测API"
api_version: "1.0.0"

# 模型配置
model_path: ./models/fire_detection.onnx
confidence_threshold: 0.25
iou_threshold: 0.45
max_detections: 1000

# 性能配置
max_file_size: 10485760 # 10MB
batch_processing: false
cache_results: true

# 安全配置
cors_origins: ["*"]
api_key_required: false
```

```bash
# 使用配置文件启动
python inference/deploy.py --config configs/deploy_config.yaml
```

### 3. 负载均衡部署

```nginx
# nginx配置文件
upstream fire_detection_backend {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
    server localhost:8004;
}

server {
    listen 80;
    server_name fire-detection.example.com;

    location / {
        proxy_pass http://fire_detection_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # 上传大小限制
        client_max_body_size 20M;

        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

```bash
# 启动多个API实例
for port in 8001 8002 8003 8004; do
    python inference/deploy.py \
        --model-path ./models/fire_detection.onnx \
        --port $port &
done
```

## 🐳 Docker 部署

### 1. 构建 Docker 镜像

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# 创建非root用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python3", "inference/deploy.py", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# 构建镜像
docker build -t fire-detection:latest .

# 运行容器
docker run -d \
    --name fire-detection \
    --gpus all \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/configs:/app/configs \
    fire-detection:latest
```

### 2. Docker Compose 部署

```yaml
# docker-compose.yml
version: "3.8"

services:
  fire-detection-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./configs:/app/configs
      - ./logs:/app/logs
    environment:
      - FIRE_DETECT_INFERENCE_DEVICE=cuda
      - FIRE_DETECT_INFERENCE_BACKEND=onnx_cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - fire-detection-api
    restart: unless-stopped

volumes:
  models:
  logs:
```

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f fire-detection-api

# 停止服务
docker-compose down
```

## ☁️ 云平台部署

### 1. AWS 部署

```yaml
# aws-deployment.yml (AWS ECS)
version: 1
applicationName: fire-detection-system
deploymentConfigName: CodeDeployDefault.ECSAllAtOnce

taskDefinition:
  family: fire-detection
  networkMode: awsvpc
  requiresCompatibilities:
    - EC2
    - FARGATE
  cpu: 2048
  memory: 4096

  containerDefinitions:
    - name: fire-detection-api
      image: your-account.dkr.ecr.region.amazonaws.com/fire-detection:latest
      portMappings:
        - containerPort: 8000
          protocol: tcp
      environment:
        - name: AWS_DEFAULT_REGION
          value: us-west-2
      logConfiguration:
        logDriver: awslogs
        options:
          awslogs-group: /ecs/fire-detection
          awslogs-region: us-west-2
          awslogs-stream-prefix: ecs
```

### 2. Azure 部署

```yaml
# azure-container-instance.yml
apiVersion: 2019-12-01
location: eastus
name: fire-detection-group
properties:
  containers:
    - name: fire-detection-api
      properties:
        image: your-registry.azurecr.io/fire-detection:latest
        ports:
          - port: 8000
            protocol: TCP
        resources:
          requests:
            cpu: 2
            memoryInGB: 4
        environmentVariables:
          - name: FIRE_DETECT_INFERENCE_DEVICE
            value: cpu
  osType: Linux
  ipAddress:
    type: Public
    ports:
      - protocol: TCP
        port: 8000
  restartPolicy: Always
```

### 3. Google Cloud 部署

```yaml
# gcp-cloud-run.yml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: fire-detection-service
  namespace: default
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containers:
        - image: gcr.io/your-project/fire-detection:latest
          ports:
            - containerPort: 8000
          resources:
            limits:
              cpu: 2000m
              memory: 4Gi
          env:
            - name: FIRE_DETECT_INFERENCE_DEVICE
              value: cpu
```

## ⚡ 硬件优化配置

### 1. NVIDIA GPU 优化

```bash
# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 环境变量优化
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1

# TensorRT优化
python -c "
from training.model_trainer import ModelTrainer
trainer = ModelTrainer()
trainer.export_model(
    export_formats=['engine'],
    trt_fp16=True,
    trt_int8=False
)
"
```

### 2. AMD GPU 优化

```bash
# ROCm安装
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.4.3/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update && sudo apt install rocm-dev rocm-libs

# 环境变量
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# DirectML优化 (Windows)
pip install onnxruntime-directml
```

### 3. 算能 AI 芯片优化

```bash
# SDK安装
tar -xzf sophon-sdk-xxx.tar.gz
cd sophon-sdk
sudo ./install.sh

# 环境配置
export SOPHON_SDK_PATH=/opt/sophon
export LD_LIBRARY_PATH=$SOPHON_SDK_PATH/lib:$LD_LIBRARY_PATH

# 模型转换
python -c "
from core.model_manager import ModelManager
mm = ModelManager()
mm.convert_to_bmodel(
    onnx_path='fire_detection.onnx',
    output_path='fire_detection_int8.bmodel',
    target='BM1684X',
    opt_level=2
)
"
```

## 📊 监控与维护

### 1. 性能监控

```python
# monitoring/performance_monitor.py
import psutil
import GPUtil
from prometheus_client import start_http_server, Gauge

# 定义指标
cpu_usage = Gauge('cpu_usage_percent', 'CPU使用率')
memory_usage = Gauge('memory_usage_percent', '内存使用率')
gpu_usage = Gauge('gpu_usage_percent', 'GPU使用率')
inference_time = Gauge('inference_time_seconds', '推理耗时')
throughput = Gauge('throughput_fps', '推理吞吐量')

def collect_metrics():
    # CPU和内存
    cpu_usage.set(psutil.cpu_percent())
    memory_usage.set(psutil.virtual_memory().percent)

    # GPU
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_usage.set(gpus[0].load * 100)

if __name__ == "__main__":
    start_http_server(9090)
    while True:
        collect_metrics()
        time.sleep(10)
```

### 2. 日志配置

```yaml
# configs/logging.yaml
version: 1
formatters:
  default:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: default
    filename: logs/fire_detection.log
    maxBytes: 10485760 # 10MB
    backupCount: 10
root:
  level: INFO
  handlers: [console, file]
loggers:
  inference:
    level: DEBUG
  training:
    level: INFO
```

### 3. 健康检查

```python
# health_check.py
import requests
import time
import logging

def health_check(endpoint="http://localhost:8000/health"):
    try:
        response = requests.get(endpoint, timeout=5)
        if response.status_code == 200:
            return True
        else:
            logging.error(f"健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"健康检查异常: {e}")
        return False

def main():
    while True:
        if not health_check():
            # 发送告警
            logging.critical("服务不可用，需要重启!")
            # TODO: 实现自动重启逻辑
        time.sleep(30)

if __name__ == "__main__":
    main()
```

### 4. 自动更新脚本

```bash
#!/bin/bash
# update_deployment.sh

set -e

echo "开始更新部署..."

# 备份当前模型
cp models/fire_detection.onnx models/fire_detection_backup_$(date +%Y%m%d_%H%M%S).onnx

# 下载新模型
wget -O models/fire_detection_new.onnx https://your-model-repo/latest/fire_detection.onnx

# 验证新模型
python -c "
from inference.fire_detector import FireDetector
detector = FireDetector('models/fire_detection_new.onnx')
print('模型验证通过')
"

# 替换模型
mv models/fire_detection_new.onnx models/fire_detection.onnx

# 重启服务
docker-compose restart fire-detection-api

echo "部署更新完成!"
```

## 🔒 安全配置

### 1. API 安全

```python
# 添加API密钥认证
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    if token.credentials != "your-secret-api-key":
        raise HTTPException(status_code=401, detail="无效的API密钥")
    return token
```

### 2. HTTPS 配置

```bash
# 生成SSL证书
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# 启动HTTPS服务
python inference/deploy.py \
    --ssl-keyfile key.pem \
    --ssl-certfile cert.pem \
    --port 443
```

## 📞 技术支持

如遇到部署问题，请提供以下信息：

- 系统环境 (OS, Python 版本, GPU 型号)
- 错误日志
- 配置文件内容
- 硬件资源使用情况

联系方式: ai@example.com
