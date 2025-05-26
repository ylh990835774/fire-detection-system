#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理器
负责模型的加载、保存、转换和版本管理
支持 PyTorch、ONNX、TensorRT 等多种模型格式
"""

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 模型相关导入
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """模型信息"""

    model_name: str
    model_path: str
    model_format: str  # pytorch/onnx/tensorrt
    model_version: str
    model_size: int  # 文件大小（字节）
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    num_classes: int
    class_names: List[str]
    training_info: Dict[str, Any]
    created_at: str
    checksum: str
    metadata: Dict[str, Any]


@dataclass
class ConversionConfig:
    """模型转换配置"""

    input_size: int = 640
    batch_size: int = 1
    dynamic_axes: bool = True
    opset_version: int = 11
    half_precision: bool = False
    simplify: bool = True


class ModelManager:
    """模型管理器"""

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)

        # 模型信息存储
        self.models_info_file = self.models_dir / "models_info.json"
        self._models_info: Dict[str, ModelInfo] = {}

        # 加载已有模型信息
        self._load_models_info()

    def _load_models_info(self):
        """加载模型信息"""
        if self.models_info_file.exists():
            try:
                with open(self.models_info_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for model_id, info_dict in data.items():
                    self._models_info[model_id] = ModelInfo(**info_dict)

                self.logger.info(f"加载了 {len(self._models_info)} 个模型信息")

            except Exception as e:
                self.logger.error(f"加载模型信息失败: {e}")

    def _save_models_info(self):
        """保存模型信息"""
        try:
            data = {}
            for model_id, model_info in self._models_info.items():
                data[model_id] = asdict(model_info)

            with open(self.models_info_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info("模型信息已保存")

        except Exception as e:
            self.logger.error(f"保存模型信息失败: {e}")

    def _calculate_checksum(self, file_path: Path) -> str:
        """计算文件校验和"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_model_id(self, model_name: str, model_version: str) -> str:
        """生成模型ID"""
        return f"{model_name}_v{model_version}"

    def register_model(
        self,
        model_name: str,
        model_path: str,
        model_format: str,
        model_version: str,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        num_classes: int,
        class_names: List[str],
        training_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """注册模型"""

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 生成模型ID
        model_id = self._get_model_id(model_name, model_version)

        # 计算文件信息
        model_size = model_path.stat().st_size
        checksum = self._calculate_checksum(model_path)

        # 复制模型文件到管理目录
        target_path = self.models_dir / f"{model_id}.{model_format}"
        if target_path != model_path:
            shutil.copy2(model_path, target_path)
            self.logger.info(f"模型文件已复制到: {target_path}")

        # 创建模型信息
        model_info = ModelInfo(
            model_name=model_name,
            model_path=str(target_path),
            model_format=model_format,
            model_version=model_version,
            model_size=model_size,
            input_shape=input_shape,
            output_shape=output_shape,
            num_classes=num_classes,
            class_names=class_names,
            training_info=training_info or {},
            created_at=datetime.now().isoformat(),
            checksum=checksum,
            metadata=metadata or {},
        )

        # 保存模型信息
        self._models_info[model_id] = model_info
        self._save_models_info()

        self.logger.info(f"模型已注册: {model_id}")
        return model_id

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        return self._models_info.get(model_id)

    def list_models(self, model_name: Optional[str] = None) -> List[ModelInfo]:
        """列出模型"""
        models = list(self._models_info.values())

        if model_name:
            models = [m for m in models if m.model_name == model_name]

        # 按创建时间排序
        models.sort(key=lambda x: x.created_at, reverse=True)
        return models

    def get_latest_model(self, model_name: str) -> Optional[ModelInfo]:
        """获取最新版本的模型"""
        models = self.list_models(model_name)
        return models[0] if models else None

    def delete_model(self, model_id: str) -> bool:
        """删除模型"""
        if model_id not in self._models_info:
            self.logger.warning(f"模型不存在: {model_id}")
            return False

        model_info = self._models_info[model_id]
        model_path = Path(model_info.model_path)

        try:
            # 删除模型文件
            if model_path.exists():
                model_path.unlink()
                self.logger.info(f"删除模型文件: {model_path}")

            # 删除模型信息
            del self._models_info[model_id]
            self._save_models_info()

            self.logger.info(f"模型已删除: {model_id}")
            return True

        except Exception as e:
            self.logger.error(f"删除模型失败 {model_id}: {e}")
            return False

    def convert_to_onnx(
        self, pytorch_model_path: str, output_path: str, config: Optional[ConversionConfig] = None
    ) -> bool:
        """将PyTorch模型转换为ONNX格式"""

        if not TORCH_AVAILABLE or not ONNX_AVAILABLE:
            raise RuntimeError("PyTorch或ONNX未安装，无法进行转换")

        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics未安装，无法进行转换")

        config = config or ConversionConfig()

        try:
            # 加载YOLO模型
            model = YOLO(pytorch_model_path)

            # 导出为ONNX
            export_path = model.export(
                format="onnx",
                imgsz=config.input_size,
                opset=config.opset_version,
                half=config.half_precision,
                dynamic=config.dynamic_axes,
                simplify=config.simplify,
            )

            # 移动到指定位置
            if export_path != output_path:
                shutil.move(export_path, output_path)

            self.logger.info(f"模型已转换为ONNX: {output_path}")

            # 验证ONNX模型
            if self._validate_onnx_model(output_path):
                return True
            else:
                self.logger.error("ONNX模型验证失败")
                return False

        except Exception as e:
            self.logger.error(f"模型转换失败: {e}")
            return False

    def _validate_onnx_model(self, onnx_path: str) -> bool:
        """验证ONNX模型"""
        try:
            # 加载并检查ONNX模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # 测试推理会话
            session = ort.InferenceSession(onnx_path)

            self.logger.info("ONNX模型验证通过")
            return True

        except Exception as e:
            self.logger.error(f"ONNX模型验证失败: {e}")
            return False

    def optimize_onnx_model(self, onnx_path: str, output_path: str) -> bool:
        """优化ONNX模型"""
        try:
            # 优化选项
            optimization_options = {
                "enable_extended_graph_optimization": True,
                "enable_graph_optimization": True,
                "graph_optimization_level": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            }

            # 创建优化的推理会话
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.optimized_model_filepath = output_path

            session = ort.InferenceSession(onnx_path, session_options)

            self.logger.info(f"ONNX模型已优化: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"ONNX模型优化失败: {e}")
            return False

    def export_model_package(self, model_id: str, output_dir: str) -> bool:
        """导出模型包（包含模型文件和元数据）"""
        if model_id not in self._models_info:
            self.logger.error(f"模型不存在: {model_id}")
            return False

        model_info = self._models_info[model_id]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 复制模型文件
            model_file = Path(model_info.model_path)
            target_model_file = output_dir / model_file.name
            shutil.copy2(model_file, target_model_file)

            # 创建元数据文件
            metadata = {
                "model_info": asdict(model_info),
                "export_time": datetime.now().isoformat(),
                "format_version": "1.0",
            }

            metadata_file = output_dir / "model_metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # 创建README文件
            readme_content = self._generate_model_readme(model_info)
            readme_file = output_dir / "README.md"
            with open(readme_file, "w", encoding="utf-8") as f:
                f.write(readme_content)

            self.logger.info(f"模型包已导出到: {output_dir}")
            return True

        except Exception as e:
            self.logger.error(f"导出模型包失败: {e}")
            return False

    def _generate_model_readme(self, model_info: ModelInfo) -> str:
        """生成模型README文档"""
        readme = f"""# {model_info.model_name} 模型

## 模型信息

- **模型名称**: {model_info.model_name}
- **版本**: {model_info.model_version}
- **格式**: {model_info.model_format}
- **输入形状**: {model_info.input_shape}
- **输出形状**: {model_info.output_shape}
- **类别数量**: {model_info.num_classes}
- **类别名称**: {", ".join(model_info.class_names)}
- **模型大小**: {model_info.model_size / (1024 * 1024):.2f} MB
- **创建时间**: {model_info.created_at}
- **校验和**: {model_info.checksum}

## 训练信息

"""

        if model_info.training_info:
            for key, value in model_info.training_info.items():
                readme += f"- **{key}**: {value}\n"

        readme += """

## 使用说明

### PyTorch模型
```python
from ultralytics import YOLO
model = YOLO('model.pt')
results = model('image.jpg')
```

### ONNX模型
```python
import onnxruntime as ort
session = ort.InferenceSession('model.onnx')
```

## 注意事项

1. 确保输入图像尺寸符合模型要求
2. 推理前进行适当的预处理
3. 根据硬件选择合适的推理后端
"""

        return readme

    def import_model_package(self, package_dir: str) -> Optional[str]:
        """导入模型包"""
        package_dir = Path(package_dir)
        metadata_file = package_dir / "model_metadata.json"

        if not metadata_file.exists():
            self.logger.error("模型包元数据文件不存在")
            return None

        try:
            # 读取元数据
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            model_info_dict = metadata["model_info"]
            model_info = ModelInfo(**model_info_dict)

            # 查找模型文件
            model_file = None
            for file in package_dir.iterdir():
                if file.suffix in [".pt", ".onnx", ".engine"]:
                    model_file = file
                    break

            if not model_file:
                self.logger.error("模型包中未找到模型文件")
                return None

            # 导入模型
            model_id = self._get_model_id(model_info.model_name, model_info.model_version)
            target_path = self.models_dir / model_file.name

            shutil.copy2(model_file, target_path)
            model_info.model_path = str(target_path)

            # 重新计算校验和
            model_info.checksum = self._calculate_checksum(target_path)

            # 保存模型信息
            self._models_info[model_id] = model_info
            self._save_models_info()

            self.logger.info(f"模型包已导入: {model_id}")
            return model_id

        except Exception as e:
            self.logger.error(f"导入模型包失败: {e}")
            return None

    def get_model_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        total_models = len(self._models_info)
        total_size = sum(info.model_size for info in self._models_info.values())

        format_stats = {}
        for info in self._models_info.values():
            format_stats[info.model_format] = format_stats.get(info.model_format, 0) + 1

        return {
            "total_models": total_models,
            "total_size_mb": total_size / (1024 * 1024),
            "format_distribution": format_stats,
            "models_dir": str(self.models_dir),
        }


# 全局模型管理器实例
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """获取模型管理器实例"""
    return model_manager


if __name__ == "__main__":
    # 测试代码
    mm = ModelManager()
    stats = mm.get_model_statistics()
    print(f"模型统计: {stats}")
