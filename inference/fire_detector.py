#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火点检测器
基于YOLO的火点检测实现
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from .detector import BaseDetector, DetectionResult, DetectorFactory

try:
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


class FireDetector(BaseDetector):
    """火点检测器"""

    def __init__(
        self,
        model_path: str,
        class_names: List[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
    ):
        """
        初始化火点检测器

        Args:
            model_path: 模型文件路径
            class_names: 类别名称列表
            confidence_threshold: 置信度阈值
            iou_threshold: IoU阈值
            input_size: 输入图像尺寸
        """
        if class_names is None:
            class_names = ["fire"]

        super().__init__(model_path, class_names, confidence_threshold, iou_threshold)

        self.input_size = input_size
        self.session = None
        self.model_format = self._detect_model_format()

        self.logger = logging.getLogger(self.__class__.__name__)

    def _detect_model_format(self) -> str:
        """检测模型格式"""
        suffix = self.model_path.suffix.lower()
        if suffix == ".onnx":
            return "onnx"
        elif suffix == ".pt":
            return "pytorch"
        elif suffix == ".engine":
            return "tensorrt"
        else:
            raise ValueError(f"不支持的模型格式: {suffix}")

    def load_model(self) -> bool:
        """加载模型"""
        try:
            if not self.validate_model_file():
                return False

            if self.model_format == "onnx":
                return self._load_onnx_model()
            elif self.model_format == "pytorch":
                return self._load_pytorch_model()
            else:
                self.logger.error(f"不支持的模型格式: {self.model_format}")
                return False

        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False

    def _load_onnx_model(self) -> bool:
        """加载ONNX模型"""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime未安装")

        try:
            # 获取可用的providers
            from core.hardware_manager import get_hardware_manager

            hw_manager = get_hardware_manager()

            providers = []
            best_hardware = hw_manager.get_best_hardware()

            if best_hardware and "cuda" in best_hardware.device_name.lower():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            # 创建推理会话
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(str(self.model_path), session_options, providers=providers)

            # 获取输入输出信息
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]

            self.logger.info(f"ONNX模型加载成功，使用providers: {self.session.get_providers()}")
            self.is_loaded = True
            return True

        except Exception as e:
            self.logger.error(f"ONNX模型加载失败: {e}")
            return False

    def _load_pytorch_model(self) -> bool:
        """加载PyTorch模型"""
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics未安装")

        try:
            self.model = YOLO(str(self.model_path))
            self.logger.info("PyTorch模型加载成功")
            self.is_loaded = True
            return True

        except Exception as e:
            self.logger.error(f"PyTorch模型加载失败: {e}")
            return False

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        # 获取原始尺寸
        original_h, original_w = image.shape[:2]

        # 保持宽高比的resize
        scale = min(self.input_size / original_w, self.input_size / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        # Resize图像
        resized = cv2.resize(image, (new_w, new_h))

        # 创建输入图像 (padding到正方形)
        input_image = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)

        # 计算padding位置
        top = (self.input_size - new_h) // 2
        left = (self.input_size - new_w) // 2

        input_image[top : top + new_h, left : left + new_w] = resized

        if self.model_format == "onnx":
            # ONNX格式需要转换为NCHW和归一化
            input_image = input_image.transpose(2, 0, 1)  # HWC -> CHW
            input_image = input_image.astype(np.float32) / 255.0  # 归一化
            input_image = np.expand_dims(input_image, axis=0)  # 添加batch维度

        # 保存预处理信息用于后处理
        self._preprocess_info = {
            "scale": scale,
            "pad_top": top,
            "pad_left": left,
            "original_shape": (original_h, original_w),
        }

        return input_image

    def _inference(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """模型推理"""
        try:
            if self.model_format == "onnx":
                # ONNX推理
                outputs = self.session.run(self.output_names, {self.input_name: preprocessed_image})
                return outputs[0]  # 返回第一个输出

            elif self.model_format == "pytorch":
                # PyTorch推理
                results = self.model(preprocessed_image, verbose=False)
                if results and len(results) > 0:
                    # 从results中提取检测数据
                    result = results[0]
                    if hasattr(result, "boxes") and result.boxes is not None:
                        boxes = result.boxes
                        # 构造输出格式 [batch, num_detections, 6] (x1, y1, x2, y2, conf, class)
                        detections = []
                        if len(boxes.xyxy) > 0:
                            for i in range(len(boxes.xyxy)):
                                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                                conf = boxes.conf[i].cpu().numpy()
                                cls = boxes.cls[i].cpu().numpy()
                                detections.append([x1, y1, x2, y2, conf, cls])

                        if detections:
                            return np.array(detections).reshape(1, -1, 6)
                        else:
                            return np.zeros((1, 1, 6))  # 空检测结果
                return np.zeros((1, 1, 6))  # 空检测结果

        except Exception as e:
            self.logger.error(f"推理失败: {e}")
            return np.zeros((1, 1, 6))  # 返回空结果

    def _postprocess(self, raw_output: np.ndarray, image_shape: Tuple[int, int]) -> DetectionResult:
        """后处理"""
        try:
            if raw_output.size == 0 or raw_output.shape[-1] < 6:
                # 空结果
                return DetectionResult(
                    boxes=np.array([]),
                    scores=np.array([]),
                    class_ids=np.array([]),
                    class_names=self.class_names,
                    image_shape=image_shape,
                )

            # 处理不同的输出格式
            if len(raw_output.shape) == 3:
                # [batch, num_detections, features]
                detections = raw_output[0]  # 取第一个batch
            else:
                detections = raw_output

            if detections.shape[-1] >= 6:
                # 格式: [x1, y1, x2, y2, conf, class] 或 [cx, cy, w, h, conf, class]
                boxes = detections[:, :4]
                scores = detections[:, 4]
                class_ids = detections[:, 5].astype(int)
            else:
                # 处理其他格式
                return DetectionResult(
                    boxes=np.array([]),
                    scores=np.array([]),
                    class_ids=np.array([]),
                    class_names=self.class_names,
                    image_shape=image_shape,
                )

            # 置信度过滤
            conf_mask = scores >= self.confidence_threshold
            boxes = boxes[conf_mask]
            scores = scores[conf_mask]
            class_ids = class_ids[conf_mask]

            if len(boxes) == 0:
                return DetectionResult(
                    boxes=np.array([]),
                    scores=np.array([]),
                    class_ids=np.array([]),
                    class_names=self.class_names,
                    image_shape=image_shape,
                )

            # 检查边界框格式并转换为xyxy
            if boxes.shape[1] == 4:
                # 检查是否为xywh格式
                if np.all(boxes[:, 2:] <= 1.0):  # width和height都小于等于1，可能是归一化的xywh
                    # 转换xywh到xyxy
                    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    boxes = np.stack([x1, y1, x2, y2], axis=1)

            # 反向映射到原始图像坐标
            if hasattr(self, "_preprocess_info"):
                info = self._preprocess_info
                scale = info["scale"]
                pad_top = info["pad_top"]
                pad_left = info["pad_left"]

                # 移除padding
                boxes[:, [0, 2]] -= pad_left
                boxes[:, [1, 3]] -= pad_top

                # 缩放回原始尺寸
                boxes /= scale

                # 如果坐标是归一化的，转换为像素坐标
                if np.all(boxes <= 1.0):
                    boxes[:, [0, 2]] *= image_shape[1]  # width
                    boxes[:, [1, 3]] *= image_shape[0]  # height

            # 边界框修正
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_shape[1])
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_shape[0])

            # NMS (非极大值抑制)
            if len(boxes) > 1:
                indices = self._nms(boxes, scores, self.iou_threshold)
                boxes = boxes[indices]
                scores = scores[indices]
                class_ids = class_ids[indices]

            return DetectionResult(
                boxes=boxes, scores=scores, class_ids=class_ids, class_names=self.class_names, image_shape=image_shape
            )

        except Exception as e:
            self.logger.error(f"后处理失败: {e}")
            return DetectionResult(
                boxes=np.array([]),
                scores=np.array([]),
                class_ids=np.array([]),
                class_names=self.class_names,
                image_shape=image_shape,
            )

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """非极大值抑制"""
        try:
            # 计算面积
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)

            # 按置信度排序
            order = scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)

                if order.size == 1:
                    break

                # 计算IoU
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                intersection = w * h

                iou = intersection / (areas[i] + areas[order[1:]] - intersection)

                # 保留IoU小于阈值的检测
                inds = np.where(iou <= iou_threshold)[0]
                order = order[inds + 1]

            return np.array(keep)

        except Exception as e:
            self.logger.error(f"NMS处理失败: {e}")
            return np.arange(len(boxes))


# 注册火点检测器
DetectorFactory.register("fire", FireDetector)


if __name__ == "__main__":
    # 测试代码
    try:
        detector = FireDetector(model_path="./models/fire_detection.onnx", confidence_threshold=0.25)
        print("火点检测器创建成功")

        # 测试模型信息
        info = detector.get_model_info()
        print(f"模型信息: {info}")

    except Exception as e:
        print(f"测试失败: {e}")
