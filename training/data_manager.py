#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管理器
负责数据集的加载、预处理、验证和增强
支持 YOLO 格式的数据集
"""

import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataManager:
    """数据管理器"""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)

        # 数据集信息
        self.dataset_info = {}
        self.class_names = []
        self.num_classes = 0

    def create_yolo_dataset(
        self,
        images_dir: str,
        labels_dir: str,
        dataset_name: str = "fire_detection",
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        class_names: Optional[List[str]] = None,
    ) -> str:
        """创建YOLO格式数据集"""

        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("数据划分比例之和必须等于1.0")

        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError("图像或标签目录不存在")

        # 创建数据集目录
        dataset_dir = self.data_root / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        # 创建子目录
        for split in ["train", "val", "test"]:
            (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # 获取所有图像文件
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = []

        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        if not image_files:
            raise ValueError("未找到任何图像文件")

        # 验证标签文件
        valid_pairs = []
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                valid_pairs.append((img_file, label_file))
            else:
                self.logger.warning(f"标签文件不存在: {label_file}")

        if not valid_pairs:
            raise ValueError("未找到任何有效的图像-标签对")

        self.logger.info(f"找到 {len(valid_pairs)} 个有效的图像-标签对")

        # 随机打乱数据
        random.shuffle(valid_pairs)

        # 计算分割点
        total_count = len(valid_pairs)
        train_count = int(total_count * train_split)
        val_count = int(total_count * val_split)

        # 分割数据
        train_pairs = valid_pairs[:train_count]
        val_pairs = valid_pairs[train_count : train_count + val_count]
        test_pairs = valid_pairs[train_count + val_count :]

        # 复制文件到对应目录
        for split_name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
            if not pairs:
                continue

            for img_file, label_file in pairs:
                # 复制图像文件
                target_img = dataset_dir / split_name / "images" / img_file.name
                shutil.copy2(img_file, target_img)

                # 复制标签文件
                target_label = dataset_dir / split_name / "labels" / label_file.name
                shutil.copy2(label_file, target_label)

        # 处理类别名称
        if class_names is None:
            class_names = ["fire"]  # 默认火点检测

        self.class_names = class_names
        self.num_classes = len(class_names)

        # 创建数据集配置文件
        config_data = {
            "path": str(dataset_dir),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": self.num_classes,
            "names": {i: name for i, name in enumerate(class_names)},
        }

        config_file = dataset_dir / "dataset.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        # 保存数据集信息
        self.dataset_info[dataset_name] = {
            "path": str(dataset_dir),
            "config_file": str(config_file),
            "train_count": len(train_pairs),
            "val_count": len(val_pairs),
            "test_count": len(test_pairs),
            "class_names": class_names,
            "num_classes": self.num_classes,
        }

        self.logger.info(f"数据集创建完成: {dataset_dir}")
        self.logger.info(f"训练集: {len(train_pairs)}, 验证集: {len(val_pairs)}, 测试集: {len(test_pairs)}")

        return str(config_file)

    def validate_dataset(self, dataset_config: str) -> Dict[str, any]:
        """验证数据集完整性"""
        config_path = Path(dataset_config)
        if not config_path.exists():
            raise FileNotFoundError(f"数据集配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        dataset_path = Path(config["path"])
        validation_results = {"valid": True, "errors": [], "warnings": [], "statistics": {}}

        # 检查目录结构
        required_dirs = ["train/images", "train/labels", "val/images", "val/labels"]
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                validation_results["errors"].append(f"缺少目录: {dir_path}")
                validation_results["valid"] = False

        if not validation_results["valid"]:
            return validation_results

        # 检查文件对应关系
        for split in ["train", "val", "test"]:
            images_dir = dataset_path / split / "images"
            labels_dir = dataset_path / split / "labels"

            if not images_dir.exists():
                continue

            image_files = list(images_dir.glob("*.*"))
            label_files = list(labels_dir.glob("*.txt"))

            # 检查图像-标签对应
            missing_labels = []
            for img_file in image_files:
                expected_label = labels_dir / f"{img_file.stem}.txt"
                if not expected_label.exists():
                    missing_labels.append(str(img_file))

            if missing_labels:
                validation_results["warnings"].extend([f"{split}集中缺少标签文件: {file}" for file in missing_labels])

            # 验证标签文件格式
            invalid_labels = []
            for label_file in label_files:
                if not self._validate_label_file(label_file, config["nc"]):
                    invalid_labels.append(str(label_file))

            if invalid_labels:
                validation_results["errors"].extend([f"标签文件格式错误: {file}" for file in invalid_labels])
                validation_results["valid"] = False

            # 统计信息
            validation_results["statistics"][split] = {
                "image_count": len(image_files),
                "label_count": len(label_files),
                "missing_labels": len(missing_labels),
                "invalid_labels": len(invalid_labels),
            }

        return validation_results

    def _validate_label_file(self, label_file: Path, num_classes: int) -> bool:
        """验证单个标签文件格式"""
        try:
            with open(label_file, "r") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    self.logger.error(f"{label_file}:{line_num} - 标签格式错误，应为5个值")
                    return False

                try:
                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= num_classes:
                        self.logger.error(f"{label_file}:{line_num} - 类别ID超出范围: {class_id}")
                        return False

                    # 检查边界框坐标
                    bbox = [float(x) for x in parts[1:]]
                    for coord in bbox:
                        if coord < 0 or coord > 1:
                            self.logger.error(f"{label_file}:{line_num} - 坐标超出范围 [0,1]: {coord}")
                            return False

                except ValueError as e:
                    self.logger.error(f"{label_file}:{line_num} - 数值解析错误: {e}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"读取标签文件失败 {label_file}: {e}")
            return False

    def get_data_augmentation(self, input_size: int = 640, training: bool = True):
        """获取数据增强管道"""
        if not ALBUMENTATIONS_AVAILABLE:
            self.logger.warning("Albumentations 未安装，将使用基础增强")
            return None

        if training:
            # 训练时的数据增强
            transforms = A.Compose(
                [
                    A.LongestMaxSize(max_size=input_size),
                    A.PadIfNeeded(min_height=input_size, min_width=input_size, border_mode=0, value=(114, 114, 114)),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.RandomRotate90(p=0.1),
                    A.OneOf(
                        [
                            A.MotionBlur(p=0.2),
                            A.MedianBlur(blur_limit=3, p=0.1),
                            A.Blur(blur_limit=3, p=0.1),
                        ],
                        p=0.2,
                    ),
                    A.OneOf(
                        [
                            A.OpticalDistortion(p=0.3),
                            A.GridDistortion(p=0.1),
                            A.PiecewiseAffine(p=0.3),
                        ],
                        p=0.2,
                    ),
                    A.OneOf(
                        [
                            A.CLAHE(clip_limit=2),
                            A.Sharpen(),
                            A.Emboss(),
                            A.RandomBrightnessContrast(),
                        ],
                        p=0.3,
                    ),
                    A.HueSaturationValue(p=0.3),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )
        else:
            # 验证/推理时的基础变换
            transforms = A.Compose(
                [
                    A.LongestMaxSize(max_size=input_size),
                    A.PadIfNeeded(min_height=input_size, min_width=input_size, border_mode=0, value=(114, 114, 114)),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )

        return transforms

    def analyze_dataset(self, dataset_config: str) -> Dict[str, any]:
        """分析数据集统计信息"""
        config_path = Path(dataset_config)
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        dataset_path = Path(config["path"])
        analysis = {
            "dataset_path": str(dataset_path),
            "num_classes": config["nc"],
            "class_names": list(config["names"].values()),
            "splits": {},
        }

        for split in ["train", "val", "test"]:
            images_dir = dataset_path / split / "images"
            labels_dir = dataset_path / split / "labels"

            if not images_dir.exists():
                continue

            image_files = list(images_dir.glob("*.*"))
            label_files = list(labels_dir.glob("*.txt"))

            # 分析标签统计
            class_counts = [0] * config["nc"]
            total_objects = 0

            for label_file in label_files:
                try:
                    with open(label_file, "r") as f:
                        lines = f.readlines()

                    for line in lines:
                        line = line.strip()
                        if line:
                            class_id = int(line.split()[0])
                            if 0 <= class_id < config["nc"]:
                                class_counts[class_id] += 1
                                total_objects += 1
                except:
                    continue

            analysis["splits"][split] = {
                "image_count": len(image_files),
                "label_count": len(label_files),
                "object_count": total_objects,
                "class_distribution": {config["names"][i]: count for i, count in enumerate(class_counts)},
            }

        return analysis

    def export_dataset_info(self, dataset_config: str, output_file: str):
        """导出数据集信息"""
        analysis = self.analyze_dataset(dataset_config)
        validation = self.validate_dataset(dataset_config)

        info = {"analysis": analysis, "validation": validation, "export_time": str(np.datetime64("now"))}

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(info, f, default_flow_style=False, allow_unicode=True)

        self.logger.info(f"数据集信息已导出到: {output_file}")


if __name__ == "__main__":
    # 测试代码
    dm = DataManager("./test_data")
    print("数据管理器初始化完成")
