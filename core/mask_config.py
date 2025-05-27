#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掩码配置系统
支持三角形区域定义和管理
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TriangularMask:
    """三角形掩码定义"""

    # 三角形三个顶点坐标 (x, y)
    vertices: List[Tuple[float, float]]

    # 掩码名称和描述
    name: str = "triangle_mask"
    description: str = ""

    # 是否启用该掩码
    enabled: bool = True

    # 坐标类型：'pixel' 或 'normalized' (0-1)
    coordinate_type: str = "pixel"

    # 掩码类型：'exclude' 排除区域, 'include' 仅检测区域
    mask_type: str = "exclude"

    # 扩展边距（像素）
    padding: int = 0

    def __post_init__(self):
        """验证输入数据"""
        if len(self.vertices) != 3:
            raise ValueError("三角形必须有且仅有3个顶点")

        if self.coordinate_type not in ["pixel", "normalized"]:
            raise ValueError("坐标类型必须是 'pixel' 或 'normalized'")

        if self.mask_type not in ["exclude", "include"]:
            raise ValueError("掩码类型必须是 'exclude' 或 'include'")

    def to_pixel_coordinates(self, image_width: int, image_height: int) -> "TriangularMask":
        """转换为像素坐标"""
        if self.coordinate_type == "pixel":
            return self

        pixel_vertices = []
        for x, y in self.vertices:
            pixel_x = x * image_width
            pixel_y = y * image_height
            pixel_vertices.append((pixel_x, pixel_y))

        return TriangularMask(
            vertices=pixel_vertices,
            name=self.name,
            description=self.description,
            enabled=self.enabled,
            coordinate_type="pixel",
            mask_type=self.mask_type,
            padding=self.padding,
        )

    def to_normalized_coordinates(self, image_width: int, image_height: int) -> "TriangularMask":
        """转换为归一化坐标"""
        if self.coordinate_type == "normalized":
            return self

        norm_vertices = []
        for x, y in self.vertices:
            norm_x = x / image_width
            norm_y = y / image_height
            norm_vertices.append((norm_x, norm_y))

        return TriangularMask(
            vertices=norm_vertices,
            name=self.name,
            description=self.description,
            enabled=self.enabled,
            coordinate_type="normalized",
            mask_type=self.mask_type,
            padding=self.padding,
        )

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """获取三角形的边界框 (x_min, y_min, x_max, y_max)"""
        xs = [x for x, y in self.vertices]
        ys = [y for x, y in self.vertices]

        x_min, x_max = min(xs) - self.padding, max(xs) + self.padding
        y_min, y_max = min(ys) - self.padding, max(ys) + self.padding

        return x_min, y_min, x_max, y_max

    def contains_point(self, x: float, y: float) -> bool:
        """判断点是否在三角形内部（重心坐标法）"""
        x1, y1 = self.vertices[0]
        x2, y2 = self.vertices[1]
        x3, y3 = self.vertices[2]

        # 计算重心坐标
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(denom) < 1e-10:  # 三角形退化
            return False

        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
        c = 1 - a - b

        # 考虑padding的影响
        threshold = -self.padding / max(abs(x2 - x1), abs(y2 - y1), 1)  # 归一化padding

        return a >= threshold and b >= threshold and c >= threshold

    def contains_box(self, x1: float, y1: float, x2: float, y2: float, overlap_threshold: float = 0.5) -> bool:
        """判断边界框是否与三角形重叠"""
        # 检查边界框的四个角点
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        inside_count = sum(1 for x, y in corners if self.contains_point(x, y))

        # 检查重叠比例
        overlap_ratio = inside_count / 4.0
        return overlap_ratio >= overlap_threshold

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "vertices": [list(vertex) for vertex in self.vertices],  # 确保元组转换为列表
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "coordinate_type": self.coordinate_type,
            "mask_type": self.mask_type,
            "padding": self.padding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TriangularMask":
        """从字典创建对象"""
        # 确保vertices是元组列表
        vertices = [tuple(vertex) for vertex in data["vertices"]]
        data_copy = data.copy()
        data_copy["vertices"] = vertices
        return cls(**data_copy)


@dataclass
class MaskConfig:
    """掩码配置类"""

    # 三角形掩码列表
    triangular_masks: List[TriangularMask] = field(default_factory=list)

    # 全局配置
    default_coordinate_type: str = "pixel"
    default_mask_type: str = "exclude"

    # 处理模式
    processing_mode: str = "hybrid"  # 'preprocess', 'postprocess', 'hybrid'

    # 掩码可视化配置
    visualization: Dict[str, Any] = field(
        default_factory=lambda: {
            "show_masks": True,
            "mask_alpha": 0.3,
            "mask_color": (255, 0, 0),  # 红色
            "line_thickness": 2,
        }
    )

    def __post_init__(self):
        """验证配置"""
        if self.processing_mode not in ["preprocess", "postprocess", "hybrid"]:
            raise ValueError("处理模式必须是 'preprocess', 'postprocess' 或 'hybrid'")

    def add_mask(self, mask: TriangularMask):
        """添加掩码"""
        self.triangular_masks.append(mask)

    def remove_mask(self, name: str) -> bool:
        """移除指定名称的掩码"""
        for i, mask in enumerate(self.triangular_masks):
            if mask.name == name:
                del self.triangular_masks[i]
                return True
        return False

    def get_mask(self, name: str) -> Optional[TriangularMask]:
        """获取指定名称的掩码"""
        for mask in self.triangular_masks:
            if mask.name == name:
                return mask
        return None

    def get_enabled_masks(self, mask_type: Optional[str] = None) -> List[TriangularMask]:
        """获取启用的掩码"""
        masks = [mask for mask in self.triangular_masks if mask.enabled]
        if mask_type:
            masks = [mask for mask in masks if mask.mask_type == mask_type]
        return masks

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "triangular_masks": [mask.to_dict() for mask in self.triangular_masks],
            "default_coordinate_type": self.default_coordinate_type,
            "default_mask_type": self.default_mask_type,
            "processing_mode": self.processing_mode,
            "visualization": self.visualization,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaskConfig":
        """从字典创建配置"""
        # 处理三角形掩码
        triangular_masks = []
        for mask_data in data.get("triangular_masks", []):
            triangular_masks.append(TriangularMask.from_dict(mask_data))

        return cls(
            triangular_masks=triangular_masks,
            default_coordinate_type=data.get("default_coordinate_type", "pixel"),
            default_mask_type=data.get("default_mask_type", "exclude"),
            processing_mode=data.get("processing_mode", "hybrid"),
            visualization=data.get("visualization", {}),
        )

    def save_to_file(self, file_path: Union[str, Path]):
        """保存配置到YAML文件"""
        file_path = Path(file_path)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True, indent=2)
            logger.info(f"掩码配置已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存掩码配置失败: {e}")
            raise

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "MaskConfig":
        """从YAML文件加载配置"""
        file_path = Path(file_path)
        try:
            if not file_path.exists():
                logger.warning(f"掩码配置文件不存在: {file_path}，使用默认配置")
                return cls()

            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning("掩码配置文件为空，使用默认配置")
                return cls()

            logger.info(f"从文件加载掩码配置: {file_path}")
            return cls.from_dict(data)

        except Exception as e:
            logger.error(f"加载掩码配置失败: {e}")
            raise


# 全局掩码配置管理器
_mask_config: Optional[MaskConfig] = None


def get_mask_config() -> MaskConfig:
    """获取全局掩码配置"""
    global _mask_config
    if _mask_config is None:
        # 尝试从默认位置加载配置
        config_path = Path(__file__).parent.parent / "configs" / "mask_config.yaml"
        try:
            _mask_config = MaskConfig.load_from_file(config_path)
        except:
            _mask_config = MaskConfig()  # 使用默认配置
    return _mask_config


def set_mask_config(config: MaskConfig):
    """设置全局掩码配置"""
    global _mask_config
    _mask_config = config


def reset_mask_config():
    """重置全局掩码配置"""
    global _mask_config
    _mask_config = None
