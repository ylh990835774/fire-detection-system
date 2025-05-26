#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理引擎
提供统一的推理接口，支持多种检测器和批处理
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .detector import BaseDetector, DetectionResult
from .fire_detector import FireDetector

logger = logging.getLogger(__name__)


class InferenceEngine:
    """推理引擎"""

    def __init__(self, detector: Optional[BaseDetector] = None, max_workers: int = 4, batch_size: int = 1):
        """
        初始化推理引擎

        Args:
            detector: 检测器实例
            max_workers: 最大工作线程数
            batch_size: 批处理大小
        """
        self.detector = detector
        self.max_workers = max_workers
        self.batch_size = batch_size

        self.logger = logging.getLogger(self.__class__.__name__)

        # 统计信息
        self.stats = {"total_inferences": 0, "total_time": 0.0, "avg_time_per_inference": 0.0, "current_fps": 0.0}

    def set_detector(self, detector: BaseDetector):
        """设置检测器"""
        self.detector = detector
        self.logger.info(f"设置检测器: {detector.__class__.__name__}")

    def detect_single(self, image: Union[str, np.ndarray], return_image: bool = False) -> Union[DetectionResult, tuple]:
        """
        单张图像检测

        Args:
            image: 图像路径或numpy数组
            return_image: 是否返回处理后的图像

        Returns:
            检测结果，可选择同时返回图像
        """
        if self.detector is None:
            raise RuntimeError("未设置检测器")

        start_time = time.time()

        try:
            result = self.detector.detect(image, return_image=return_image)

            # 更新统计信息
            inference_time = time.time() - start_time
            self._update_stats(inference_time)

            return result

        except Exception as e:
            self.logger.error(f"单张图像检测失败: {e}")
            raise

    def detect_batch(self, images: List[Union[str, np.ndarray]], use_threading: bool = True) -> List[DetectionResult]:
        """
        批量检测

        Args:
            images: 图像列表
            use_threading: 是否使用多线程

        Returns:
            检测结果列表
        """
        if self.detector is None:
            raise RuntimeError("未设置检测器")

        if not images:
            return []

        start_time = time.time()

        try:
            if use_threading and len(images) > 1:
                results = self._detect_batch_threaded(images)
            else:
                results = self._detect_batch_sequential(images)

            # 更新统计信息
            total_time = time.time() - start_time
            avg_time = total_time / len(images)

            for _ in images:
                self._update_stats(avg_time)

            return results

        except Exception as e:
            self.logger.error(f"批量检测失败: {e}")
            raise

    def _detect_batch_sequential(self, images: List[Union[str, np.ndarray]]) -> List[DetectionResult]:
        """顺序批量检测"""
        results = []

        for i, image in enumerate(images):
            try:
                result = self.detector.detect(image)
                results.append(result)

                self.logger.debug(f"完成检测 {i + 1}/{len(images)}")

            except Exception as e:
                self.logger.error(f"检测图像 {i} 失败: {e}")
                # 创建空结果
                empty_result = DetectionResult(
                    boxes=np.array([]),
                    scores=np.array([]),
                    class_ids=np.array([]),
                    class_names=self.detector.class_names,
                    image_shape=(640, 640),  # 默认尺寸
                )
                results.append(empty_result)

        return results

    def _detect_batch_threaded(self, images: List[Union[str, np.ndarray]]) -> List[DetectionResult]:
        """多线程批量检测"""
        results = [None] * len(images)

        def detect_single_threaded(index_image_pair):
            index, image = index_image_pair
            try:
                result = self.detector.detect(image)
                return index, result
            except Exception as e:
                self.logger.error(f"检测图像 {index} 失败: {e}")
                # 创建空结果
                empty_result = DetectionResult(
                    boxes=np.array([]),
                    scores=np.array([]),
                    class_ids=np.array([]),
                    class_names=self.detector.class_names,
                    image_shape=(640, 640),
                )
                return index, empty_result

        # 使用线程池
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = executor.map(detect_single_threaded, enumerate(images))

            for index, result in futures:
                results[index] = result

        return results

    def detect_video_stream(
        self,
        video_source: Union[str, int],
        frame_skip: int = 0,
        max_frames: Optional[int] = None,
        save_results: bool = False,
        output_dir: Optional[str] = None,
    ) -> List[DetectionResult]:
        """
        视频流检测

        Args:
            video_source: 视频文件路径或摄像头索引
            frame_skip: 跳帧数量
            max_frames: 最大处理帧数
            save_results: 是否保存结果
            output_dir: 输出目录

        Returns:
            检测结果列表
        """
        if self.detector is None:
            raise RuntimeError("未设置检测器")

        try:
            import cv2
        except ImportError:
            raise RuntimeError("需要安装OpenCV进行视频处理")

        # 打开视频
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {video_source}")

        results = []
        frame_count = 0
        processed_count = 0

        # 创建输出目录
        if save_results and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # 跳帧处理
                if frame_skip > 0 and (frame_count - 1) % (frame_skip + 1) != 0:
                    continue

                # 检查最大帧数限制
                if max_frames and processed_count >= max_frames:
                    break

                try:
                    # 转换BGR到RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 检测
                    result = self.detector.detect(rgb_frame)
                    results.append(result)

                    processed_count += 1

                    # 保存结果
                    if save_results and output_dir:
                        self._save_detection_result(frame, result, output_path / f"frame_{processed_count:06d}.jpg")

                    self.logger.debug(f"处理帧 {processed_count}, 检测到 {result.num_detections} 个目标")

                except Exception as e:
                    self.logger.error(f"处理帧 {frame_count} 失败: {e}")
                    continue

        finally:
            cap.release()

        self.logger.info(f"视频处理完成: 总帧数 {frame_count}, 处理帧数 {processed_count}")
        return results

    def _save_detection_result(self, image: np.ndarray, result: DetectionResult, output_path: Path):
        """保存检测结果"""
        try:
            import cv2

            # 绘制检测框
            output_image = image.copy()

            for i in range(result.num_detections):
                box = result.boxes[i]
                score = result.scores[i]
                label = result.get_labels()[i]

                # 绘制边界框
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制标签
                text = f"{label}: {score:.2f}"
                cv2.putText(output_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 保存图像
            cv2.imwrite(str(output_path), output_image)

        except Exception as e:
            self.logger.error(f"保存检测结果失败: {e}")

    def _update_stats(self, inference_time: float):
        """更新统计信息"""
        self.stats["total_inferences"] += 1
        self.stats["total_time"] += inference_time
        self.stats["avg_time_per_inference"] = self.stats["total_time"] / self.stats["total_inferences"]
        self.stats["current_fps"] = 1.0 / inference_time if inference_time > 0 else 0.0

    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        return self.stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {"total_inferences": 0, "total_time": 0.0, "avg_time_per_inference": 0.0, "current_fps": 0.0}
        self.logger.info("统计信息已重置")

    def benchmark(
        self, test_images: List[Union[str, np.ndarray]], num_runs: int = 10, warmup_runs: int = 3
    ) -> Dict[str, float]:
        """
        性能基准测试

        Args:
            test_images: 测试图像列表
            num_runs: 测试轮数
            warmup_runs: 预热轮数

        Returns:
            性能统计信息
        """
        if self.detector is None:
            raise RuntimeError("未设置检测器")

        if not test_images:
            raise ValueError("测试图像列表不能为空")

        self.logger.info(f"开始性能基准测试: {num_runs} 轮测试, {warmup_runs} 轮预热")

        # 预热
        for i in range(warmup_runs):
            for image in test_images:
                self.detector.detect(image)

        # 正式测试
        times = []
        for run in range(num_runs):
            start_time = time.time()

            for image in test_images:
                self.detector.detect(image)

            run_time = time.time() - start_time
            times.append(run_time)

            self.logger.debug(f"测试轮 {run + 1}: {run_time:.3f}s")

        # 计算统计信息
        total_inferences = num_runs * len(test_images)
        total_time = sum(times)
        avg_time_per_batch = np.mean(times)
        avg_time_per_image = total_time / total_inferences
        fps = total_inferences / total_time

        benchmark_results = {
            "total_inferences": total_inferences,
            "total_time": total_time,
            "avg_time_per_batch": avg_time_per_batch,
            "avg_time_per_image": avg_time_per_image,
            "fps": fps,
            "min_batch_time": min(times),
            "max_batch_time": max(times),
            "std_batch_time": np.std(times),
            "batch_size": len(test_images),
        }

        self.logger.info(f"基准测试完成: FPS={fps:.1f}, 平均推理时间={avg_time_per_image * 1000:.1f}ms")

        return benchmark_results


def create_fire_detection_engine(
    model_path: str, confidence_threshold: float = 0.25, iou_threshold: float = 0.45, max_workers: int = 4
) -> InferenceEngine:
    """
    创建火点检测推理引擎

    Args:
        model_path: 模型文件路径
        confidence_threshold: 置信度阈值
        iou_threshold: IoU阈值
        max_workers: 最大工作线程数

    Returns:
        配置好的推理引擎
    """
    # 创建火点检测器
    detector = FireDetector(
        model_path=model_path, confidence_threshold=confidence_threshold, iou_threshold=iou_threshold
    )

    # 创建推理引擎
    engine = InferenceEngine(detector=detector, max_workers=max_workers)

    return engine


if __name__ == "__main__":
    # 测试代码
    try:
        # 创建推理引擎
        engine = create_fire_detection_engine(model_path="./models/fire_detection.onnx")

        print("推理引擎创建成功")
        print(f"统计信息: {engine.get_stats()}")

    except Exception as e:
        print(f"测试失败: {e}")
