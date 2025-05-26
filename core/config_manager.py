#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理器
统一管理训练、推理和部署相关的配置参数
支持 YAML 配置文件、环境变量和运行时参数覆盖
"""

import logging
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""

    # 模型配置
    model_name: str = "yolov8n"  # yolov8n/s/m/l/x
    input_size: int = 640
    num_classes: int = 1  # 火点检测为1类
    class_names: List[str] = None

    # 训练参数
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    momentum: float = 0.937

    # 数据配置
    data_path: str = "datasets/fire_detection"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # 增强配置
    augmentation: bool = True
    mixup: float = 0.0
    copy_paste: float = 0.0

    # 保存配置
    save_dir: str = "runs/train"
    save_period: int = 10

    # 硬件配置
    device: str = "auto"  # auto/cpu/cuda/mps
    workers: int = 8

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ["fire"]


@dataclass
class InferenceConfig:
    """推理配置"""

    # 模型配置
    model_path: str = "models/fire_detection.pt"
    model_format: str = "onnx"  # onnx/pytorch/tensorrt
    input_size: int = 640

    # 推理参数
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 1000

    # 硬件配置
    device: str = "auto"
    backend: str = "auto"  # auto/onnxruntime/tensorrt/openvino

    # 输出配置
    save_results: bool = True
    save_dir: str = "runs/inference"
    show_labels: bool = True
    show_confidence: bool = True
    line_thickness: int = 3


@dataclass
class ModelConfig:
    """模型配置"""

    # 基础配置
    architecture: str = "yolov8"
    variant: str = "n"  # n/s/m/l/x
    num_classes: int = 1

    # 网络结构
    depth_multiple: float = 0.33
    width_multiple: float = 0.25
    max_channels: int = 1024

    # 锚框配置
    anchors: Optional[List[List[float]]] = None

    # 损失函数配置
    box_loss_gain: float = 0.05
    cls_loss_gain: float = 0.5
    dfl_loss_gain: float = 1.5


@dataclass
class DeployConfig:
    """部署配置"""

    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # API配置
    api_title: str = "火点检测API"
    api_version: str = "1.0.0"
    docs_url: str = "/docs"

    # 安全配置
    cors_origins: List[str] = None
    api_key_required: bool = False

    # 性能配置
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    batch_processing: bool = False
    cache_results: bool = True

    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)

        # 配置缓存
        self._config_cache: Dict[str, Any] = {}

        # 默认配置
        self._default_configs = {
            "training": TrainingConfig(),
            "inference": InferenceConfig(),
            "model": ModelConfig(),
            "deploy": DeployConfig(),
        }

    def load_config(self, config_type: str, config_file: Optional[str] = None) -> Dict[str, Any]:
        """加载配置文件"""
        if config_file is None:
            config_file = f"{config_type}_config.yaml"

        config_path = self.config_dir / config_file

        # 检查缓存
        cache_key = f"{config_type}:{config_path}"
        if cache_key in self._config_cache:
            return deepcopy(self._config_cache[cache_key])

        # 获取默认配置
        default_config = self._default_configs.get(config_type)
        if default_config is None:
            raise ValueError(f"未知的配置类型: {config_type}")

        config_dict = asdict(default_config)

        # 从文件加载配置
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    file_config = yaml.safe_load(f) or {}

                # 合并配置
                config_dict.update(file_config)
                self.logger.info(f"从 {config_path} 加载配置")

            except Exception as e:
                self.logger.error(f"加载配置文件失败 {config_path}: {e}")
        else:
            self.logger.info(f"配置文件 {config_path} 不存在，使用默认配置")

        # 环境变量覆盖
        config_dict = self._apply_env_overrides(config_dict, config_type)

        # 缓存配置
        self._config_cache[cache_key] = deepcopy(config_dict)

        return config_dict

    def save_config(self, config_type: str, config_dict: Dict[str, Any], config_file: Optional[str] = None):
        """保存配置到文件"""
        if config_file is None:
            config_file = f"{config_type}_config.yaml"

        config_path = self.config_dir / config_file

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)

            self.logger.info(f"配置已保存到 {config_path}")

            # 更新缓存
            cache_key = f"{config_type}:{config_path}"
            self._config_cache[cache_key] = deepcopy(config_dict)

        except Exception as e:
            self.logger.error(f"保存配置文件失败 {config_path}: {e}")
            raise

    def get_training_config(self, config_file: Optional[str] = None) -> TrainingConfig:
        """获取训练配置"""
        config_dict = self.load_config("training", config_file)
        return TrainingConfig(**config_dict)

    def get_inference_config(self, config_file: Optional[str] = None) -> InferenceConfig:
        """获取推理配置"""
        config_dict = self.load_config("inference", config_file)
        return InferenceConfig(**config_dict)

    def get_model_config(self, config_file: Optional[str] = None) -> ModelConfig:
        """获取模型配置"""
        config_dict = self.load_config("model", config_file)
        return ModelConfig(**config_dict)

    def get_deploy_config(self, config_file: Optional[str] = None) -> DeployConfig:
        """获取部署配置"""
        config_dict = self.load_config("deploy", config_file)
        return DeployConfig(**config_dict)

    def update_config(self, config_type: str, updates: Dict[str, Any], config_file: Optional[str] = None):
        """更新配置"""
        config_dict = self.load_config(config_type, config_file)
        config_dict.update(updates)
        self.save_config(config_type, config_dict, config_file)

    def _apply_env_overrides(self, config_dict: Dict[str, Any], config_type: str) -> Dict[str, Any]:
        """应用环境变量覆盖"""
        prefix = f"FIRE_DETECT_{config_type.upper()}_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix) :].lower()
                if config_key in config_dict:
                    # 尝试转换类型
                    original_value = config_dict[config_key]
                    try:
                        if isinstance(original_value, bool):
                            config_dict[config_key] = value.lower() in ("true", "1", "yes", "on")
                        elif isinstance(original_value, int):
                            config_dict[config_key] = int(value)
                        elif isinstance(original_value, float):
                            config_dict[config_key] = float(value)
                        elif isinstance(original_value, list):
                            config_dict[config_key] = value.split(",")
                        else:
                            config_dict[config_key] = value

                        self.logger.info(f"环境变量覆盖: {config_key} = {config_dict[config_key]}")

                    except ValueError as e:
                        self.logger.warning(f"环境变量类型转换失败 {key}: {e}")

        return config_dict

    def create_default_configs(self):
        """创建默认配置文件"""
        for config_type, default_config in self._default_configs.items():
            config_file = f"{config_type}_config.yaml"
            config_path = self.config_dir / config_file

            if not config_path.exists():
                config_dict = asdict(default_config)
                self.save_config(config_type, config_dict, config_file)
                self.logger.info(f"创建默认配置文件: {config_path}")

    def validate_config(self, config_type: str, config_dict: Dict[str, Any]) -> bool:
        """验证配置有效性"""
        try:
            if config_type == "training":
                TrainingConfig(**config_dict)
            elif config_type == "inference":
                InferenceConfig(**config_dict)
            elif config_type == "model":
                ModelConfig(**config_dict)
            elif config_type == "deploy":
                DeployConfig(**config_dict)
            else:
                return False

            return True

        except Exception as e:
            self.logger.error(f"配置验证失败: {e}")
            return False

    def clear_cache(self):
        """清除配置缓存"""
        self._config_cache.clear()
        self.logger.info("配置缓存已清除")


# 全局配置管理器实例
config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """获取配置管理器实例"""
    return config_manager


if __name__ == "__main__":
    # 测试代码
    cm = ConfigManager()
    cm.create_default_configs()

    # 测试各种配置
    train_config = cm.get_training_config()
    print(f"训练配置: {train_config}")

    inference_config = cm.get_inference_config()
    print(f"推理配置: {inference_config}")
