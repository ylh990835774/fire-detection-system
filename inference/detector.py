#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测器基类
提供统一的检测接口，支持插件化扩展
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class DetectionResult:
    """检测结果类"""

    def __init__(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        class_names: List[str],
        image_shape: Tuple[int, int],
    ):
        """
        Args:
            boxes: 边界框 [[x1, y1, x2, y2], ...]
            scores: 置信度分数 [score1, score2, ...]
            class_ids: 类别ID [id1, id2, ...]
            class_names: 类别名称列表
            image_shape: 原始图像形状 (height, width)
        """
        self.boxes = boxes
        self.scores = scores
        self.class_ids = class_ids
        self.class_names = class_names
        self.image_shape = image_shape

    @property
    def num_detections(self) -> int:
        """检测到的目标数量"""
        return len(self.boxes)

    def get_labels(self) -> List[str]:
        """获取检测目标的标签"""
        return [self.class_names[class_id] for class_id in self.class_ids]

    def filter_by_confidence(self, min_confidence: float) -> "DetectionResult":
        """根据置信度过滤结果"""
        mask = self.scores >= min_confidence
        return DetectionResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            class_ids=self.class_ids[mask],
            class_names=self.class_names,
            image_shape=self.image_shape,
        )

    def filter_by_class(self, target_classes: List[str]) -> "DetectionResult":
        """根据类别过滤结果"""
        target_ids = [self.class_names.index(cls) for cls in target_classes if cls in self.class_names]
        mask = np.isin(self.class_ids, target_ids)
        return DetectionResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            class_ids=self.class_ids[mask],
            class_names=self.class_names,
            image_shape=self.image_shape,
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "num_detections": self.num_detections,
            "boxes": self.boxes.tolist(),
            "scores": self.scores.tolist(),
            "class_ids": self.class_ids.tolist(),
            "labels": self.get_labels(),
            "image_shape": self.image_shape,
        }


class BaseDetector(ABC):
    """检测器基类"""

    def __init__(
        self, model_path: str, class_names: List[str], confidence_threshold: float = 0.25, iou_threshold: float = 0.45
    ):
        """
        Args:
            model_path: 模型文件路径
            class_names: 类别名称列表
            confidence_threshold: 置信度阈值
            iou_threshold: IoU阈值
        """
        self.model_path = Path(model_path)
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.logger = logging.getLogger(self.__class__.__name__)

        # 模型相关属性
        self.model = None
        self.input_size = None
        self.is_loaded = False

        # 预处理和后处理函数
        self._preprocess_fn = None
        self._postprocess_fn = None

    @abstractmethod
    def load_model(self) -> bool:
        """加载模型"""
        pass

    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        pass

    @abstractmethod
    def _inference(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """模型推理"""
        pass

    @abstractmethod
    def _postprocess(self, raw_output: np.ndarray, image_shape: Tuple[int, int]) -> DetectionResult:
        """后处理"""
        pass

    def detect(
        self, image: Union[str, np.ndarray], return_image: bool = False
    ) -> Union[DetectionResult, Tuple[DetectionResult, np.ndarray]]:
        """
        检测图像中的目标

        Args:
            image: 图像路径或numpy数组
            return_image: 是否返回处理后的图像

        Returns:
            检测结果，可选择同时返回图像
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("模型加载失败")

        # 加载图像
        if isinstance(image, str):
            image = self._load_image(image)
        elif not isinstance(image, np.ndarray):
            raise ValueError("图像必须是文件路径或numpy数组")

        original_shape = image.shape[:2]  # (height, width)

        # 预处理
        preprocessed = self._preprocess(image)

        # 推理
        raw_output = self._inference(preprocessed)

        # 后处理
        result = self._postprocess(raw_output, original_shape)

        if return_image:
            return result, image
        else:
            return result

    def detect_batch(self, images: List[Union[str, np.ndarray]], batch_size: int = 1) -> List[DetectionResult]:
        """批量检测"""
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            batch_results = []

            for image in batch:
                result = self.detect(image)
                batch_results.append(result)

            results.extend(batch_results)

        return results

    def _load_image(self, image_path: str) -> np.ndarray:
        """加载图像文件"""
        from PIL import Image

        try:
            image = Image.open(image_path).convert("RGB")
            return np.array(image)
        except Exception as e:
            self.logger.error(f"加载图像失败 {image_path}: {e}")
            raise

    def set_thresholds(self, confidence: Optional[float] = None, iou: Optional[float] = None):
        """设置检测阈值"""
        if confidence is not None:
            self.confidence_threshold = confidence
        if iou is not None:
            self.iou_threshold = iou

        self.logger.info(f"阈值已更新: confidence={self.confidence_threshold}, iou={self.iou_threshold}")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_path": str(self.model_path),
            "class_names": self.class_names,
            "num_classes": len(self.class_names),
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "input_size": self.input_size,
            "is_loaded": self.is_loaded,
        }

    def validate_model_file(self) -> bool:
        """验证模型文件是否存在且有效"""
        if not self.model_path.exists():
            self.logger.error(f"模型文件不存在: {self.model_path}")
            return False

        if self.model_path.stat().st_size == 0:
            self.logger.error(f"模型文件为空: {self.model_path}")
            return False

        return True

    def benchmark(self, test_images: List[Union[str, np.ndarray]], num_runs: int = 10) -> Dict[str, float]:
        """性能基准测试"""
        import time

        if not test_images:
            raise ValueError("测试图像列表不能为空")

        if not self.is_loaded:
            self.load_model()

        # 预热
        for _ in range(3):
            self.detect(test_images[0])

        # 测试
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            for image in test_images:
                self.detect(image)
            end_time = time.time()
            times.append(end_time - start_time)

        total_inferences = num_runs * len(test_images)
        avg_time = np.mean(times)
        fps = total_inferences / sum(times)

        return {
            "avg_time_per_batch": avg_time,
            "avg_time_per_image": avg_time / len(test_images),
            "fps": fps,
            "total_inferences": total_inferences,
            "batch_size": len(test_images),
        }


class DetectorFactory:
    """检测器工厂类"""

    _detectors = {}

    @classmethod
    def register(cls, name: str, detector_class: type):
        """注册检测器"""
        cls._detectors[name] = detector_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseDetector:
        """创建检测器实例"""
        if name not in cls._detectors:
            raise ValueError(f"未知的检测器类型: {name}")

        return cls._detectors[name](**kwargs)

    @classmethod
    def list_detectors(cls) -> List[str]:
        """列出所有注册的检测器"""
        return list(cls._detectors.keys())
