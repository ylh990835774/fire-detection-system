#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掩码管理器
处理三角形区域掩码的生成、管理和应用
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .mask_config import MaskConfig, TriangularMask, get_mask_config

logger = logging.getLogger(__name__)


class MaskManager:
    """掩码管理器"""

    def __init__(self, config: Optional[MaskConfig] = None):
        """
        初始化掩码管理器

        Args:
            config: 掩码配置，如果为None则使用全局配置
        """
        self.config = config or get_mask_config()
        self.logger = logging.getLogger(self.__class__.__name__)

        # 缓存生成的掩码
        self._mask_cache: Dict[str, np.ndarray] = {}
        self._cache_image_shape: Optional[Tuple[int, int]] = None

    def clear_cache(self):
        """清空掩码缓存"""
        self._mask_cache.clear()
        self._cache_image_shape = None
        self.logger.debug("掩码缓存已清空")

    def generate_triangle_mask(self, mask: TriangularMask, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        生成单个三角形掩码

        Args:
            mask: 三角形掩码定义
            image_shape: 图像形状 (height, width)

        Returns:
            二值掩码数组，True表示掩码区域
        """
        height, width = image_shape

        # 转换为像素坐标
        pixel_mask = mask.to_pixel_coordinates(width, height)

        # 创建空的掩码
        mask_array = np.zeros((height, width), dtype=np.uint8)

        # 获取三角形顶点
        vertices = np.array(pixel_mask.vertices, dtype=np.int32)

        # 填充三角形
        cv2.fillPoly(mask_array, [vertices], 255)

        # 应用padding
        if pixel_mask.padding > 0:
            kernel = np.ones((pixel_mask.padding * 2 + 1, pixel_mask.padding * 2 + 1), np.uint8)
            mask_array = cv2.dilate(mask_array, kernel, iterations=1)

        return mask_array > 0

    def generate_combined_mask(self, image_shape: Tuple[int, int], mask_type: str = "exclude") -> np.ndarray:
        """
        生成组合掩码

        Args:
            image_shape: 图像形状 (height, width)
            mask_type: 掩码类型 'exclude' 或 'include'

        Returns:
            组合掩码数组
        """
        # 检查缓存
        cache_key = f"{image_shape}_{mask_type}"
        if cache_key in self._mask_cache and self._cache_image_shape == image_shape:
            return self._mask_cache[cache_key]

        height, width = image_shape

        # 获取指定类型的启用掩码
        masks = self.config.get_enabled_masks(mask_type)

        if not masks:
            # 没有掩码时返回全False（不遮挡任何区域）
            combined_mask = np.zeros((height, width), dtype=bool)
        else:
            # 初始化组合掩码
            combined_mask = np.zeros((height, width), dtype=bool)

            # 合并所有掩码
            for mask in masks:
                triangle_mask = self.generate_triangle_mask(mask, image_shape)
                combined_mask = np.logical_or(combined_mask, triangle_mask)

        # 缓存结果
        self._mask_cache[cache_key] = combined_mask
        self._cache_image_shape = image_shape

        return combined_mask

    def apply_mask_to_image(
        self, image: np.ndarray, mask_type: str = "exclude", fill_value: Union[int, Tuple[int, int, int]] = 0
    ) -> np.ndarray:
        """
        将掩码应用到图像上

        Args:
            image: 输入图像
            mask_type: 掩码类型
            fill_value: 掩码区域的填充值

        Returns:
            应用掩码后的图像
        """
        if len(image.shape) != 3:
            raise ValueError("图像必须是3维数组 (H, W, C)")

        height, width = image.shape[:2]
        mask = self.generate_combined_mask((height, width), mask_type)

        # 复制图像
        masked_image = image.copy()

        if mask_type == "exclude":
            # 排除区域：将掩码区域设为填充值
            if isinstance(fill_value, int):
                masked_image[mask] = fill_value
            else:
                for c in range(image.shape[2]):
                    masked_image[mask, c] = fill_value[c] if c < len(fill_value) else fill_value[-1]
        elif mask_type == "include":
            # 包含区域：只保留掩码区域
            inverse_mask = ~mask
            if isinstance(fill_value, int):
                masked_image[inverse_mask] = fill_value
            else:
                for c in range(image.shape[2]):
                    masked_image[inverse_mask, c] = fill_value[c] if c < len(fill_value) else fill_value[-1]

        return masked_image

    def filter_detections_by_mask(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        image_shape: Tuple[int, int],
        overlap_threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        根据掩码过滤检测结果

        Args:
            boxes: 检测框 [[x1, y1, x2, y2], ...]
            scores: 置信度分数
            class_ids: 类别ID
            image_shape: 图像形状
            overlap_threshold: 重叠阈值

        Returns:
            过滤后的 (boxes, scores, class_ids)
        """
        if len(boxes) == 0:
            return boxes, scores, class_ids

        # 获取排除掩码
        exclude_masks = self.config.get_enabled_masks("exclude")

        if not exclude_masks:
            return boxes, scores, class_ids

        # 过滤检测结果
        keep_indices = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            should_keep = True

            # 检查是否与任何排除掩码重叠
            for mask in exclude_masks:
                pixel_mask = mask.to_pixel_coordinates(image_shape[1], image_shape[0])
                if pixel_mask.contains_box(x1, y1, x2, y2, overlap_threshold):
                    should_keep = False
                    break

            if should_keep:
                keep_indices.append(i)

        # 应用过滤
        if keep_indices:
            filtered_boxes = boxes[keep_indices]
            filtered_scores = scores[keep_indices]
            filtered_class_ids = class_ids[keep_indices]
        else:
            filtered_boxes = np.array([])
            filtered_scores = np.array([])
            filtered_class_ids = np.array([])

        self.logger.debug(f"掩码过滤: {len(boxes)} -> {len(filtered_boxes)} 个检测结果")

        return filtered_boxes, filtered_scores, filtered_class_ids

    def visualize_masks(self, image: np.ndarray, show_exclude: bool = True, show_include: bool = True) -> np.ndarray:
        """
        在图像上可视化掩码区域

        Args:
            image: 输入图像
            show_exclude: 是否显示排除掩码
            show_include: 是否显示包含掩码

        Returns:
            添加了掩码可视化的图像
        """
        vis_image = image.copy()
        height, width = image.shape[:2]

        vis_config = self.config.visualization
        alpha = vis_config.get("mask_alpha", 0.3)
        thickness = vis_config.get("line_thickness", 2)

        # 显示排除掩码（红色）
        if show_exclude:
            exclude_masks = self.config.get_enabled_masks("exclude")
            for mask in exclude_masks:
                pixel_mask = mask.to_pixel_coordinates(width, height)
                vertices = np.array(pixel_mask.vertices, dtype=np.int32)

                # 绘制填充的三角形
                overlay = vis_image.copy()
                cv2.fillPoly(overlay, [vertices], (0, 0, 255))  # 红色
                vis_image = cv2.addWeighted(vis_image, 1 - alpha, overlay, alpha, 0)

                # 绘制边界线
                cv2.polylines(vis_image, [vertices], True, (0, 0, 255), thickness)

                # 添加标签
                center_x = int(np.mean(vertices[:, 0]))
                center_y = int(np.mean(vertices[:, 1]))
                cv2.putText(
                    vis_image,
                    f"排除: {mask.name}",
                    (center_x - 50, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        # 显示包含掩码（绿色）
        if show_include:
            include_masks = self.config.get_enabled_masks("include")
            for mask in include_masks:
                pixel_mask = mask.to_pixel_coordinates(width, height)
                vertices = np.array(pixel_mask.vertices, dtype=np.int32)

                # 绘制填充的三角形
                overlay = vis_image.copy()
                cv2.fillPoly(overlay, [vertices], (0, 255, 0))  # 绿色
                vis_image = cv2.addWeighted(vis_image, 1 - alpha, overlay, alpha, 0)

                # 绘制边界线
                cv2.polylines(vis_image, [vertices], True, (0, 255, 0), thickness)

                # 添加标签
                center_x = int(np.mean(vertices[:, 0]))
                center_y = int(np.mean(vertices[:, 1]))
                cv2.putText(
                    vis_image,
                    f"包含: {mask.name}",
                    (center_x - 50, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        return vis_image

    def add_triangular_mask(
        self,
        vertices: List[Tuple[float, float]],
        name: str,
        mask_type: str = "exclude",
        coordinate_type: str = "pixel",
        **kwargs,
    ) -> bool:
        """
        添加三角形掩码

        Args:
            vertices: 三角形顶点
            name: 掩码名称
            mask_type: 掩码类型
            coordinate_type: 坐标类型
            **kwargs: 其他参数

        Returns:
            是否添加成功
        """
        try:
            mask = TriangularMask(
                vertices=vertices, name=name, mask_type=mask_type, coordinate_type=coordinate_type, **kwargs
            )
            self.config.add_mask(mask)
            self.clear_cache()  # 清空缓存
            self.logger.info(f"添加三角形掩码: {name}")
            return True
        except Exception as e:
            self.logger.error(f"添加掩码失败: {e}")
            return False

    def remove_mask(self, name: str) -> bool:
        """移除掩码"""
        success = self.config.remove_mask(name)
        if success:
            self.clear_cache()
            self.logger.info(f"移除掩码: {name}")
        return success

    def get_mask_info(self) -> Dict[str, Any]:
        """获取掩码信息"""
        exclude_masks = self.config.get_enabled_masks("exclude")
        include_masks = self.config.get_enabled_masks("include")

        return {
            "total_masks": len(self.config.triangular_masks),
            "enabled_masks": len(exclude_masks) + len(include_masks),
            "exclude_masks": len(exclude_masks),
            "include_masks": len(include_masks),
            "processing_mode": self.config.processing_mode,
            "mask_names": [mask.name for mask in self.config.triangular_masks],
        }


# 全局掩码管理器
_mask_manager: Optional[MaskManager] = None


def get_mask_manager() -> MaskManager:
    """获取全局掩码管理器"""
    global _mask_manager
    if _mask_manager is None:
        _mask_manager = MaskManager()
    return _mask_manager


def set_mask_manager(manager: MaskManager):
    """设置全局掩码管理器"""
    global _mask_manager
    _mask_manager = manager


def reset_mask_manager():
    """重置全局掩码管理器"""
    global _mask_manager
    _mask_manager = None
