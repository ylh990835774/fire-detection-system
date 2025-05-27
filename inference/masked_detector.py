#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持掩码的检测器基类
扩展BaseDetector以支持三角形区域排除功能
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.mask_config import MaskConfig
from core.mask_manager import MaskManager, get_mask_manager

from .detector import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class MaskedDetector(BaseDetector):
    """支持掩码的检测器基类"""

    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        mask_manager: Optional[MaskManager] = None,
        enable_mask: bool = True,
    ):
        """
        初始化支持掩码的检测器

        Args:
            model_path: 模型文件路径
            class_names: 类别名称列表
            confidence_threshold: 置信度阈值
            iou_threshold: IoU阈值
            mask_manager: 掩码管理器，如果为None则使用全局管理器
            enable_mask: 是否启用掩码功能
        """
        super().__init__(model_path, class_names, confidence_threshold, iou_threshold)

        self.mask_manager = mask_manager or get_mask_manager()
        self.enable_mask = enable_mask

        self.logger = logging.getLogger(self.__class__.__name__)

    def detect(
        self,
        image: Union[str, np.ndarray],
        return_image: bool = False,
        apply_mask: bool = None,
        visualize_mask: bool = False,
    ) -> Union[DetectionResult, Tuple[DetectionResult, np.ndarray]]:
        """
        检测图像中的目标（支持掩码）

        Args:
            image: 图像路径或numpy数组
            return_image: 是否返回处理后的图像
            apply_mask: 是否应用掩码，None时使用实例设置
            visualize_mask: 是否在返回的图像上可视化掩码

        Returns:
            检测结果，可选择同时返回图像
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("模型加载失败")

        # 确定是否应用掩码
        if apply_mask is None:
            apply_mask = self.enable_mask

        # 加载图像
        if isinstance(image, str):
            original_image = self._load_image(image)
        elif isinstance(image, np.ndarray):
            original_image = image.copy()
        else:
            raise ValueError("图像必须是文件路径或numpy数组")

        original_shape = original_image.shape[:2]  # (height, width)
        processed_image = original_image.copy()

        # 预处理阶段应用掩码
        if apply_mask and self.mask_manager.config.processing_mode in ["preprocess", "hybrid"]:
            try:
                processed_image = self.mask_manager.apply_mask_to_image(
                    processed_image,
                    mask_type="exclude",
                    fill_value=0,  # 用黑色填充排除区域
                )
                self.logger.debug("已在预处理阶段应用掩码")
            except Exception as e:
                self.logger.warning(f"预处理掩码应用失败: {e}")

        # 预处理
        preprocessed = self._preprocess(processed_image)

        # 推理
        raw_output = self._inference(preprocessed)

        # 后处理
        result = self._postprocess(raw_output, original_shape)

        # 后处理阶段过滤掩码
        if apply_mask and self.mask_manager.config.processing_mode in ["postprocess", "hybrid"]:
            try:
                filtered_boxes, filtered_scores, filtered_class_ids = self.mask_manager.filter_detections_by_mask(
                    result.boxes, result.scores, result.class_ids, original_shape, overlap_threshold=0.5
                )

                # 创建过滤后的结果
                result = DetectionResult(
                    boxes=filtered_boxes,
                    scores=filtered_scores,
                    class_ids=filtered_class_ids,
                    class_names=result.class_names,
                    image_shape=result.image_shape,
                )
                self.logger.debug("已在后处理阶段过滤掩码")
            except Exception as e:
                self.logger.warning(f"后处理掩码过滤失败: {e}")

        # 处理返回结果
        if return_image:
            return_img = original_image.copy()

            # 可视化掩码
            if visualize_mask and apply_mask:
                try:
                    return_img = self.mask_manager.visualize_masks(return_img)
                except Exception as e:
                    self.logger.warning(f"掩码可视化失败: {e}")

            return result, return_img
        else:
            return result

    def detect_with_mask_visualization(self, image: Union[str, np.ndarray]) -> Tuple[DetectionResult, np.ndarray]:
        """
        检测并返回带有掩码可视化的图像

        Args:
            image: 输入图像

        Returns:
            (检测结果, 可视化图像)
        """
        return self.detect(image, return_image=True, visualize_mask=True)

    def set_mask_config(self, config: MaskConfig):
        """设置掩码配置"""
        self.mask_manager.config = config
        self.mask_manager.clear_cache()
        self.logger.info("掩码配置已更新")

    def enable_mask_feature(self, enable: bool = True):
        """启用/禁用掩码功能"""
        self.enable_mask = enable
        self.logger.info(f"掩码功能已{'启用' if enable else '禁用'}")

    def add_exclude_triangle(
        self, vertices: List[Tuple[float, float]], name: str, coordinate_type: str = "pixel", **kwargs
    ) -> bool:
        """
        添加排除三角形区域

        Args:
            vertices: 三角形顶点
            name: 区域名称
            coordinate_type: 坐标类型
            **kwargs: 其他参数

        Returns:
            是否添加成功
        """
        return self.mask_manager.add_triangular_mask(
            vertices=vertices, name=name, mask_type="exclude", coordinate_type=coordinate_type, **kwargs
        )

    def add_include_triangle(
        self, vertices: List[Tuple[float, float]], name: str, coordinate_type: str = "pixel", **kwargs
    ) -> bool:
        """
        添加包含三角形区域

        Args:
            vertices: 三角形顶点
            name: 区域名称
            coordinate_type: 坐标类型
            **kwargs: 其他参数

        Returns:
            是否添加成功
        """
        return self.mask_manager.add_triangular_mask(
            vertices=vertices, name=name, mask_type="include", coordinate_type=coordinate_type, **kwargs
        )

    def remove_mask_region(self, name: str) -> bool:
        """
        移除掩码区域

        Args:
            name: 区域名称

        Returns:
            是否移除成功
        """
        return self.mask_manager.remove_mask(name)

    def list_mask_regions(self) -> List[str]:
        """获取所有掩码区域名称"""
        return [mask.name for mask in self.mask_manager.config.triangular_masks]

    def get_mask_info(self) -> dict:
        """获取掩码信息"""
        return self.mask_manager.get_mask_info()

    def benchmark_with_mask(self, test_images: List[Union[str, np.ndarray]], num_runs: int = 10) -> dict:
        """
        包含掩码功能的性能基准测试

        Args:
            test_images: 测试图像列表
            num_runs: 测试运行次数

        Returns:
            性能指标字典
        """
        import time

        if not test_images:
            raise ValueError("测试图像列表不能为空")

        # 测试不启用掩码的性能
        self.enable_mask_feature(False)
        start_time = time.time()

        for _ in range(num_runs):
            for image in test_images:
                _ = self.detect(image)

        no_mask_time = time.time() - start_time

        # 测试启用掩码的性能
        self.enable_mask_feature(True)
        start_time = time.time()

        for _ in range(num_runs):
            for image in test_images:
                _ = self.detect(image)

        with_mask_time = time.time() - start_time

        # 计算性能指标
        total_detections = num_runs * len(test_images)

        results = {
            "total_detections": total_detections,
            "no_mask_total_time": no_mask_time,
            "with_mask_total_time": with_mask_time,
            "no_mask_avg_time": no_mask_time / total_detections,
            "with_mask_avg_time": with_mask_time / total_detections,
            "mask_overhead": with_mask_time - no_mask_time,
            "mask_overhead_percent": ((with_mask_time - no_mask_time) / no_mask_time) * 100,
            "mask_info": self.get_mask_info(),
        }

        return results

    def validate_mask_setup(self) -> dict:
        """
        验证掩码设置

        Returns:
            验证结果字典
        """
        validation_results = {
            "mask_manager_loaded": self.mask_manager is not None,
            "mask_feature_enabled": self.enable_mask,
            "total_masks": 0,
            "enabled_masks": 0,
            "exclude_masks": 0,
            "include_masks": 0,
            "processing_mode": None,
            "validation_errors": [],
        }

        if self.mask_manager:
            try:
                mask_info = self.mask_manager.get_mask_info()
                validation_results.update(mask_info)
                validation_results["processing_mode"] = self.mask_manager.config.processing_mode

                # 验证每个掩码
                for mask in self.mask_manager.config.triangular_masks:
                    try:
                        # 验证三角形有效性
                        from utils.mask_utils import validate_triangle

                        if not validate_triangle(mask.vertices):
                            validation_results["validation_errors"].append(f"无效的三角形: {mask.name}")
                    except Exception as e:
                        validation_results["validation_errors"].append(f"掩码验证错误 {mask.name}: {str(e)}")

            except Exception as e:
                validation_results["validation_errors"].append(f"掩码信息获取失败: {str(e)}")

        validation_results["is_valid"] = len(validation_results["validation_errors"]) == 0

        return validation_results
