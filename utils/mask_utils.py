#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掩码工具函数
提供三角形计算、掩码生成等实用功能
"""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


def point_in_triangle(x: float, y: float, triangle: List[Tuple[float, float]]) -> bool:
    """
    判断点是否在三角形内部（重心坐标法）

    Args:
        x, y: 点坐标
        triangle: 三角形顶点列表 [(x1,y1), (x2,y2), (x3,y3)]

    Returns:
        是否在三角形内部
    """
    x1, y1 = triangle[0]
    x2, y2 = triangle[1]
    x3, y3 = triangle[2]

    # 计算重心坐标
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-10:  # 三角形退化
        return False

    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    c = 1 - a - b

    return a >= 0 and b >= 0 and c >= 0


def triangle_area(triangle: List[Tuple[float, float]]) -> float:
    """
    计算三角形面积

    Args:
        triangle: 三角形顶点列表

    Returns:
        三角形面积
    """
    x1, y1 = triangle[0]
    x2, y2 = triangle[1]
    x3, y3 = triangle[2]

    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def triangle_centroid(triangle: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    计算三角形重心

    Args:
        triangle: 三角形顶点列表

    Returns:
        重心坐标 (x, y)
    """
    x_center = sum(x for x, y in triangle) / 3
    y_center = sum(y for x, y in triangle) / 3
    return x_center, y_center


def triangle_bounding_box(triangle: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """
    计算三角形边界框

    Args:
        triangle: 三角形顶点列表

    Returns:
        边界框 (x_min, y_min, x_max, y_max)
    """
    xs = [x for x, y in triangle]
    ys = [y for x, y in triangle]

    return min(xs), min(ys), max(xs), max(ys)


def expand_triangle(triangle: List[Tuple[float, float]], padding: float) -> List[Tuple[float, float]]:
    """
    扩展三角形（向外扩展）

    Args:
        triangle: 原始三角形顶点
        padding: 扩展距离

    Returns:
        扩展后的三角形顶点
    """
    if padding == 0:
        return triangle

    # 计算重心
    cx, cy = triangle_centroid(triangle)

    # 向外扩展每个顶点
    expanded_triangle = []
    for x, y in triangle:
        # 计算从重心到顶点的向量
        dx = x - cx
        dy = y - cy

        # 计算距离
        distance = math.sqrt(dx * dx + dy * dy)
        if distance > 0:
            # 归一化并扩展
            scale = (distance + padding) / distance
            new_x = cx + dx * scale
            new_y = cy + dy * scale
            expanded_triangle.append((new_x, new_y))
        else:
            expanded_triangle.append((x, y))

    return expanded_triangle


def box_triangle_overlap(
    box: Tuple[float, float, float, float], triangle: List[Tuple[float, float]], overlap_threshold: float = 0.5
) -> bool:
    """
    判断边界框与三角形是否重叠

    Args:
        box: 边界框 (x1, y1, x2, y2)
        triangle: 三角形顶点列表
        overlap_threshold: 重叠阈值

    Returns:
        是否重叠
    """
    x1, y1, x2, y2 = box

    # 检查边界框的四个角点
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    inside_count = sum(1 for x, y in corners if point_in_triangle(x, y, triangle))

    # 检查重叠比例
    overlap_ratio = inside_count / 4.0
    return overlap_ratio >= overlap_threshold


def normalize_coordinates(
    coordinates: List[Tuple[float, float]], image_width: int, image_height: int
) -> List[Tuple[float, float]]:
    """
    将像素坐标转换为归一化坐标

    Args:
        coordinates: 像素坐标列表
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        归一化坐标列表
    """
    normalized = []
    for x, y in coordinates:
        norm_x = x / image_width
        norm_y = y / image_height
        normalized.append((norm_x, norm_y))
    return normalized


def denormalize_coordinates(
    coordinates: List[Tuple[float, float]], image_width: int, image_height: int
) -> List[Tuple[float, float]]:
    """
    将归一化坐标转换为像素坐标

    Args:
        coordinates: 归一化坐标列表
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        像素坐标列表
    """
    pixel_coords = []
    for x, y in coordinates:
        pixel_x = x * image_width
        pixel_y = y * image_height
        pixel_coords.append((pixel_x, pixel_y))
    return pixel_coords


def create_triangle_mask(
    image_shape: Tuple[int, int], triangle: List[Tuple[float, float]], fill_value: int = 255
) -> np.ndarray:
    """
    创建三角形掩码

    Args:
        image_shape: 图像形状 (height, width)
        triangle: 三角形顶点列表
        fill_value: 填充值

    Returns:
        掩码数组
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # 转换为整数坐标
    vertices = np.array(triangle, dtype=np.int32)

    # 填充三角形
    cv2.fillPoly(mask, [vertices], fill_value)

    return mask


def apply_morphology(mask: np.ndarray, operation: str, kernel_size: int) -> np.ndarray:
    """
    对掩码应用形态学操作

    Args:
        mask: 输入掩码
        operation: 操作类型 ('dilate', 'erode', 'open', 'close')
        kernel_size: 内核大小

    Returns:
        处理后的掩码
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if operation == "dilate":
        return cv2.dilate(mask, kernel, iterations=1)
    elif operation == "erode":
        return cv2.erode(mask, kernel, iterations=1)
    elif operation == "open":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError(f"不支持的形态学操作: {operation}")


def combine_masks(masks: List[np.ndarray], operation: str = "union") -> np.ndarray:
    """
    组合多个掩码

    Args:
        masks: 掩码列表
        operation: 组合操作 ('union', 'intersection', 'difference')

    Returns:
        组合后的掩码
    """
    if not masks:
        return np.array([])

    if len(masks) == 1:
        return masks[0]

    result = masks[0].copy()

    for mask in masks[1:]:
        if operation == "union":
            result = np.logical_or(result, mask)
        elif operation == "intersection":
            result = np.logical_and(result, mask)
        elif operation == "difference":
            result = np.logical_and(result, ~mask)
        else:
            raise ValueError(f"不支持的组合操作: {operation}")

    return result.astype(np.uint8) * 255


def visualize_triangle_on_image(
    image: np.ndarray,
    triangle: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    alpha: float = 0.3,
    label: Optional[str] = None,
) -> np.ndarray:
    """
    在图像上可视化三角形

    Args:
        image: 输入图像
        triangle: 三角形顶点
        color: 颜色 (B, G, R)
        thickness: 线条粗细
        alpha: 透明度
        label: 标签文本

    Returns:
        可视化后的图像
    """
    vis_image = image.copy()
    vertices = np.array(triangle, dtype=np.int32)

    # 绘制填充的三角形
    overlay = vis_image.copy()
    cv2.fillPoly(overlay, [vertices], color)
    vis_image = cv2.addWeighted(vis_image, 1 - alpha, overlay, alpha, 0)

    # 绘制边界线
    cv2.polylines(vis_image, [vertices], True, color, thickness)

    # 添加标签
    if label:
        center_x = int(np.mean(vertices[:, 0]))
        center_y = int(np.mean(vertices[:, 1]))
        cv2.putText(
            vis_image, label, (center_x - len(label) * 4, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return vis_image


def validate_triangle(triangle: List[Tuple[float, float]]) -> bool:
    """
    验证三角形是否有效

    Args:
        triangle: 三角形顶点列表

    Returns:
        是否为有效三角形
    """
    if len(triangle) != 3:
        return False

    # 检查是否有重复点
    for i in range(3):
        for j in range(i + 1, 3):
            if triangle[i] == triangle[j]:
                return False

    # 检查三点是否共线
    area = triangle_area(triangle)
    return area > 1e-6  # 面积要大于极小值


def auto_generate_triangle_masks(
    image_shape: Tuple[int, int], num_masks: int = 4, mask_type: str = "corner"
) -> List[List[Tuple[float, float]]]:
    """
    自动生成三角形掩码

    Args:
        image_shape: 图像形状 (height, width)
        num_masks: 掩码数量
        mask_type: 掩码类型 ('corner', 'edge', 'random')

    Returns:
        三角形顶点列表
    """
    height, width = image_shape
    triangles = []

    if mask_type == "corner":
        # 四个角落的三角形
        margin = min(width, height) * 0.15

        # 左上角
        triangles.append([(0, 0), (margin, 0), (0, margin)])

        # 右上角
        triangles.append([(width - margin, 0), (width, 0), (width, margin)])

        # 右下角
        triangles.append([(width, height - margin), (width, height), (width - margin, height)])

        # 左下角
        triangles.append([(0, height - margin), (margin, height), (0, height)])

    elif mask_type == "edge":
        # 边缘中心的三角形
        margin = min(width, height) * 0.1

        # 上边缘
        triangles.append([(width * 0.4, 0), (width * 0.6, 0), (width * 0.5, margin)])

        # 右边缘
        triangles.append([(width, height * 0.4), (width, height * 0.6), (width - margin, height * 0.5)])

        # 下边缘
        triangles.append([(width * 0.4, height), (width * 0.6, height), (width * 0.5, height - margin)])

        # 左边缘
        triangles.append([(0, height * 0.4), (0, height * 0.6), (margin, height * 0.5)])

    elif mask_type == "random":
        # 随机生成三角形
        import random

        for _ in range(num_masks):
            triangle = []
            for _ in range(3):
                x = random.uniform(0, width)
                y = random.uniform(0, height)
                triangle.append((x, y))

            if validate_triangle(triangle):
                triangles.append(triangle)

    return triangles[:num_masks]
