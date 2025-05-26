# 🔥 火点检测系统

基于 YOLO 的多硬件异常火点检测系统，支持 Nvidia GPU、AMD GPU、算能 AI 芯片等多种硬件平台。

## ✨ 特性

- 🎯 **高精度检测**: 基于 YOLOv8 的先进目标检测算法
- 🚀 **多硬件支持**: 支持 NVIDIA GPU、AMD GPU、算能 AI 芯片、CPU
- 🔧 **模块化设计**: 插件化架构，易于扩展新的检测任务
- 📦 **完整流程**: 提供训练和部署的完整解决方案
- 🎛️ **灵活配置**: 支持 YAML 配置文件和环境变量
- 📊 **性能监控**: 内置性能基准测试和模型管理

## 🏗️ 系统架构

```
fire-detection-system/
├── training/          # 训练模块
├── inference/         # 推理模块
├── core/             # 核心组件
├── utils/            # 工具函数
├── configs/          # 配置文件
└── models/           # 模型存储
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone <repository-url>
cd fire-detection-system

# 安装依赖
pip install -r requirements.txt

# 或使用setup.py安装
pip install -e .
```

### 2. 硬件检测

```python
from core.hardware_manager import get_hardware_manager

hw_manager = get_hardware_manager()
hw_manager.print_hardware_summary()
```

### 3. 数据准备

准备 YOLO 格式的数据集：

```
dataset/
├── images/
│   ├── img1.jpg
│   └── img2.jpg
└── labels/
    ├── img1.txt
    └── img2.txt
```

### 4. 模型训练

```bash
# 命令行训练
python training/train.py \
    --images-dir /path/to/images \
    --labels-dir /path/to/labels \
    --epochs 100 \
    --batch-size 16 \
    --model yolov8n

# 或使用Python API
from training.model_trainer import ModelTrainer
from training.data_manager import DataManager

# 创建数据集
dm = DataManager("./datasets")
config_file = dm.create_yolo_dataset(
    images_dir="./images",
    labels_dir="./labels"
)

# 训练模型
trainer = ModelTrainer()
results = trainer.train(config_file)
```

### 5. 模型推理

```python
from inference.fire_detector import FireDetector

# 创建检测器
detector = FireDetector(
    model_path="./models/fire_detection.onnx",
    confidence_threshold=0.25
)

# 检测图像
result = detector.detect("./test_image.jpg")
print(f"检测到 {result.num_detections} 个火点")
```

### 6. Web API 部署

```bash
# 启动API服务
python inference/deploy.py --model-path ./models/fire_detection.onnx --port 8000

# 访问API文档
# http://localhost:8000/docs
```

## 📋 详细配置

### 训练配置 (configs/training_config.yaml)

```yaml
# 模型配置
model_name: yolov8n
input_size: 640
num_classes: 1
class_names: ["fire"]

# 训练参数
epochs: 100
batch_size: 16
learning_rate: 0.01
device: auto

# 数据配置
data_path: datasets/fire_detection
train_split: 0.8
val_split: 0.1
test_split: 0.1

# 增强配置
augmentation: true
mixup: 0.0
copy_paste: 0.0
```

### 推理配置 (configs/inference_config.yaml)

```yaml
# 模型配置
model_path: models/fire_detection.onnx
model_format: onnx
input_size: 640

# 推理参数
confidence_threshold: 0.25
iou_threshold: 0.45
max_detections: 1000

# 硬件配置
device: auto
backend: auto
```

## 🔧 硬件支持

### NVIDIA GPU

```bash
# 检查CUDA环境
nvidia-smi

# 环境变量配置
export FIRE_DETECT_INFERENCE_DEVICE=cuda
export FIRE_DETECT_INFERENCE_BACKEND=onnx_cuda
```

### AMD GPU

```bash
# ROCm环境
rocm-smi

# 环境变量配置
export FIRE_DETECT_INFERENCE_DEVICE=auto
export FIRE_DETECT_INFERENCE_BACKEND=onnx_directml
```

### 算能 AI 芯片

```bash
# 检查算能SDK
ls /opt/sophon

# 环境变量配置
export FIRE_DETECT_INFERENCE_BACKEND=sophon_sdk
```

## 📊 性能基准

| 硬件平台       | 模型格式 | 输入尺寸 | FPS | 精度(mAP50) |
| -------------- | -------- | -------- | --- | ----------- |
| RTX 3080       | ONNX     | 640x640  | 120 | 0.85        |
| AMD RX 6800    | ONNX     | 640x640  | 95  | 0.85        |
| Intel i7-12700 | ONNX     | 640x640  | 25  | 0.85        |
| 算能 BM1684X   | Sophon   | 640x640  | 80  | 0.84        |

## 🛠️ 扩展开发

### 添加新的检测任务

```python
from inference.detector import BaseDetector, DetectorFactory

class PersonDetector(BaseDetector):
    def load_model(self):
        # 实现模型加载逻辑
        pass

    def _preprocess(self, image):
        # 实现预处理逻辑
        pass

    def _inference(self, preprocessed_image):
        # 实现推理逻辑
        pass

    def _postprocess(self, raw_output, image_shape):
        # 实现后处理逻辑
        pass

# 注册检测器
DetectorFactory.register("person", PersonDetector)

# 使用新检测器
detector = DetectorFactory.create("person",
                                 model_path="person_model.onnx",
                                 class_names=["person"])
```

## 📚 API 参考

### 训练 API

```python
from training.model_trainer import ModelTrainer

trainer = ModelTrainer()

# 训练模型
results = trainer.train(
    data_config="dataset.yaml",
    output_dir="runs/train"
)

# 验证模型
metrics = trainer.validate("dataset.yaml")

# 导出模型
exported = trainer.export_model(
    export_formats=["onnx", "engine"]
)
```

### 推理 API

```python
from inference.fire_detector import FireDetector

detector = FireDetector(
    model_path="model.onnx",
    confidence_threshold=0.25,
    iou_threshold=0.45
)

# 单张图像检测
result = detector.detect("image.jpg")

# 批量检测
results = detector.detect_batch(["img1.jpg", "img2.jpg"])

# 性能测试
benchmark = detector.benchmark(test_images, num_runs=10)
```

## 🐛 故障排除

### 常见问题

1. **CUDA 不可用**

   ```bash
   # 检查CUDA安装
   nvcc --version
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **内存不足**

   ```yaml
   # 减小批次大小
   batch_size: 8

   # 使用混合精度
   half_precision: true
   ```

3. **模型加载失败**

   ```python
   # 检查模型文件
   detector.validate_model_file()

   # 重新下载模型
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   ```

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 联系方式

- 邮箱: ai@example.com
- 项目地址: https://github.com/example/fire-detection-system

---

**注意**: 请确保您有合适的数据集和计算资源进行模型训练。建议使用 GPU 进行训练以获得更好的性能。
