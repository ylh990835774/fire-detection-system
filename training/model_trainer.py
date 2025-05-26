#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练器
基于 YOLOv8 实现火点检测模型的训练
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import torch
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# 使用绝对导入
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config_manager import TrainingConfig, get_config_manager
from core.hardware_manager import get_hardware_manager
from core.model_manager import get_model_manager

logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器"""

    def __init__(self, config: Optional[TrainingConfig] = None):
        if not YOLO_AVAILABLE:
            raise RuntimeError("Ultralytics YOLO 未安装，无法进行训练")

        self.config = config or get_config_manager().get_training_config()
        self.hardware_manager = get_hardware_manager()
        self.model_manager = get_model_manager()

        self.logger = logging.getLogger(self.__class__.__name__)

        # 训练状态
        self.model = None
        self.training_results = None
        self.best_model_path = None

        # 回调函数
        self.callbacks = {"on_epoch_end": [], "on_train_end": [], "on_validation_end": []}

    def setup_device(self):
        """设置训练设备"""
        if self.config.device == "auto":
            # 自动选择最佳设备
            best_hardware = self.hardware_manager.get_best_hardware()
            if best_hardware:
                if "cuda" in best_hardware.device_name.lower() or "gpu" in best_hardware.device_name.lower():
                    device = "cuda"
                else:
                    device = "cpu"
            else:
                device = "cpu"
        else:
            device = self.config.device

        self.logger.info(f"使用设备: {device}")
        return device

    def create_model(self, model_name: Optional[str] = None) -> YOLO:
        """创建或加载模型"""
        model_name = model_name or self.config.model_name

        try:
            # 尝试加载预训练模型
            if model_name.endswith(".pt"):
                # 自定义模型路径
                model = YOLO(model_name)
                self.logger.info(f"加载自定义模型: {model_name}")
            else:
                # 标准YOLO模型
                model = YOLO(f"{model_name}.pt")
                self.logger.info(f"加载预训练模型: {model_name}")

            return model

        except Exception as e:
            self.logger.error(f"模型创建失败: {e}")
            raise

    def train(
        self, data_config: str, output_dir: Optional[str] = None, resume: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """训练模型"""

        # 设置输出目录
        if output_dir is None:
            output_dir = self.config.save_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 设置设备
        device = self.setup_device()

        # 创建模型
        self.model = self.create_model()

        # 训练参数
        train_args = {
            "data": data_config,
            "epochs": self.config.epochs,
            "imgsz": self.config.input_size,
            "batch": self.config.batch_size,
            "lr0": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "momentum": self.config.momentum,
            "device": device,
            "workers": self.config.workers,
            "project": str(output_dir.parent),
            "name": output_dir.name,
            "save_period": self.config.save_period,
            "augment": self.config.augmentation,
            "mixup": self.config.mixup,
            "copy_paste": self.config.copy_paste,
            "resume": resume,
            "verbose": True,
        }

        # 覆盖用户自定义参数
        train_args.update(kwargs)

        self.logger.info("开始训练...")
        self.logger.info(f"训练参数: {train_args}")

        try:
            # 记录训练开始时间
            start_time = time.time()

            # 开始训练
            results = self.model.train(**train_args)

            # 记录训练结束时间
            end_time = time.time()
            training_time = end_time - start_time

            # 保存训练结果
            self.training_results = results
            self.best_model_path = self.model.trainer.best

            self.logger.info(f"训练完成，耗时: {training_time:.2f}秒")
            self.logger.info(f"最佳模型保存在: {self.best_model_path}")

            # 生成训练报告
            training_info = {
                "training_time": training_time,
                "epochs_completed": self.config.epochs,
                "best_model_path": str(self.best_model_path),
                "device_used": device,
                "final_metrics": self._extract_final_metrics(results),
                "config": self.config.__dict__,
            }

            # 触发训练结束回调
            self._trigger_callbacks("on_train_end", training_info)

            return training_info

        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {e}")
            raise

    def validate(self, data_config: str, model_path: Optional[str] = None) -> Dict[str, Any]:
        """验证模型"""
        model_path = model_path or self.best_model_path

        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError("验证模型文件不存在")

        # 加载模型
        model = YOLO(model_path)

        # 运行验证
        self.logger.info("开始模型验证...")

        try:
            val_results = model.val(
                data=data_config,
                imgsz=self.config.input_size,
                batch=self.config.batch_size,
                device=self.setup_device(),
                verbose=True,
            )

            # 提取验证指标
            metrics = self._extract_validation_metrics(val_results)

            self.logger.info("模型验证完成")
            self.logger.info(f"验证指标: {metrics}")

            # 触发验证结束回调
            self._trigger_callbacks("on_validation_end", metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"验证过程中出现错误: {e}")
            raise

    def export_model(
        self, model_path: Optional[str] = None, export_formats: Optional[list] = None, output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """导出模型为不同格式"""

        model_path = model_path or self.best_model_path
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError("导出模型文件不存在")

        if export_formats is None:
            export_formats = ["onnx"]  # 默认导出ONNX格式

        # 加载模型
        model = YOLO(model_path)

        exported_models = {}

        for format_name in export_formats:
            try:
                self.logger.info(f"导出模型为 {format_name} 格式...")

                export_path = model.export(
                    format=format_name,
                    imgsz=self.config.input_size,
                    half=False,  # 保持FP32精度
                    dynamic=True,  # 支持动态输入
                    simplify=True,  # 简化ONNX图
                )

                exported_models[format_name] = str(export_path)
                self.logger.info(f"{format_name} 模型已导出: {export_path}")

                # 注册模型到模型管理器
                if format_name == "onnx":
                    self._register_exported_model(export_path, format_name)

            except Exception as e:
                self.logger.error(f"导出 {format_name} 格式失败: {e}")

        return exported_models

    def _register_exported_model(self, model_path: str, model_format: str):
        """注册导出的模型到模型管理器"""
        try:
            # 生成模型版本
            import datetime

            version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # 注册模型
            model_id = self.model_manager.register_model(
                model_name="fire_detection",
                model_path=model_path,
                model_format=model_format,
                model_version=version,
                input_shape=(1, 3, self.config.input_size, self.config.input_size),
                output_shape=(1, -1, self.config.num_classes + 5),
                num_classes=self.config.num_classes,
                class_names=self.config.class_names,
                training_info=self.training_results.__dict__ if self.training_results else {},
                metadata={"training_config": self.config.__dict__, "framework": "YOLOv8"},
            )

            self.logger.info(f"模型已注册到管理器: {model_id}")

        except Exception as e:
            self.logger.error(f"模型注册失败: {e}")

    def _extract_final_metrics(self, results) -> Dict[str, float]:
        """提取最终训练指标"""
        try:
            if hasattr(results, "results_dict"):
                metrics = results.results_dict
            else:
                # 从results对象中提取指标
                metrics = {}
                if hasattr(results, "maps"):
                    metrics["mAP50"] = float(results.maps[0]) if results.maps else 0.0
                    metrics["mAP50-95"] = float(results.maps[1]) if len(results.maps) > 1 else 0.0

            return metrics

        except Exception as e:
            self.logger.warning(f"无法提取训练指标: {e}")
            return {}

    def _extract_validation_metrics(self, val_results) -> Dict[str, float]:
        """提取验证指标"""
        try:
            metrics = {}

            if hasattr(val_results, "maps"):
                metrics["mAP50"] = float(val_results.maps[0]) if val_results.maps else 0.0
                metrics["mAP50-95"] = float(val_results.maps[1]) if len(val_results.maps) > 1 else 0.0

            if hasattr(val_results, "mp"):
                metrics["precision"] = float(val_results.mp)

            if hasattr(val_results, "mr"):
                metrics["recall"] = float(val_results.mr)

            if hasattr(val_results, "f1"):
                metrics["f1"] = float(val_results.f1)

            return metrics

        except Exception as e:
            self.logger.warning(f"无法提取验证指标: {e}")
            return {}

    def add_callback(self, event: str, callback: Callable):
        """添加回调函数"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            self.logger.warning(f"未知的回调事件: {event}")

    def _trigger_callbacks(self, event: str, data: Any):
        """触发回调函数"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"回调函数执行失败: {e}")

    def resume_training(self, checkpoint_path: str, **kwargs) -> Dict[str, Any]:
        """从检查点恢复训练"""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        self.logger.info(f"从检查点恢复训练: {checkpoint_path}")

        # 加载检查点
        self.model = YOLO(checkpoint_path)

        # 恢复训练
        return self.train(resume=True, **kwargs)

    def get_training_history(self) -> Dict[str, list]:
        """获取训练历史"""
        if self.training_results is None:
            return {}

        try:
            # 从训练结果中提取历史数据
            history = {}
            if hasattr(self.training_results, "history"):
                history = self.training_results.history

            return history

        except Exception as e:
            self.logger.error(f"获取训练历史失败: {e}")
            return {}


if __name__ == "__main__":
    # 测试代码
    trainer = ModelTrainer()
    print("模型训练器初始化完成")
