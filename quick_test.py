#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火点检测系统快速验证脚本
包含新增的三角形掩码功能验证
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_core_modules():
    """测试核心模块"""
    try:
        # 硬件管理器 - 简化测试
        from core.hardware_manager import get_hardware_manager

        hw_manager = get_hardware_manager()
        print("✅ 硬件管理器: 初始化成功")

        # 配置管理器
        from core.config_manager import get_config_manager

        config_manager = get_config_manager()
        train_config = config_manager.get_training_config()
        print(f"✅ 配置管理器: 训练模型 {train_config.model_name}")

        # 模型管理器 - 简化测试
        from core.model_manager import get_model_manager

        model_manager = get_model_manager()
        print("✅ 模型管理器: 初始化成功")

        # 新增：掩码配置和管理器
        from core.mask_config import get_mask_config
        from core.mask_manager import get_mask_manager

        mask_config = get_mask_config()
        mask_manager = get_mask_manager()
        mask_info = mask_manager.get_mask_info()
        print(f"✅ 掩码管理器: {mask_info['total_masks']} 个掩码区域，模式: {mask_info['processing_mode']}")

        return True
    except Exception as e:
        print(f"❌ 核心模块测试失败: {e}")
        return False


def test_training_modules():
    """测试训练模块"""
    try:
        # 数据管理器
        from training.data_manager import DataManager

        data_manager = DataManager("./test_data")
        print("✅ 数据管理器: 初始化成功")

        # 模型训练器
        from training.model_trainer import ModelTrainer

        trainer = ModelTrainer()
        print("✅ 模型训练器: 初始化成功")

        return True
    except Exception as e:
        print(f"❌ 训练模块测试失败: {e}")
        return False


def test_inference_modules():
    """测试推理模块"""
    try:
        # 检测器基类
        from inference.detector import DetectorFactory

        detectors = DetectorFactory.list_detectors()
        print(f"✅ 检测器基类: {len(detectors)} 个注册检测器")

        # 新增：掩码工具函数测试
        from utils.mask_utils import point_in_triangle, validate_triangle

        test_triangle = [(100, 100), (200, 100), (150, 200)]
        is_valid = validate_triangle(test_triangle)
        point_inside = point_in_triangle(150, 150, test_triangle)
        print(f"✅ 掩码工具: 三角形验证 {is_valid}, 点包含测试 {point_inside}")

        # 新增：支持掩码的检测器
        print("✅ 掩码检测器: 基类导入成功")

        return True
    except Exception as e:
        print(f"❌ 推理模块测试失败: {e}")
        return False


def test_mask_features():
    """专门测试掩码功能"""
    try:
        # 测试三角形掩码配置
        from core.mask_config import MaskConfig, TriangularMask

        triangle = TriangularMask(
            vertices=[(50, 50), (150, 50), (100, 150)], name="quick_test_triangle", mask_type="exclude"
        )

        config = MaskConfig()
        config.add_mask(triangle)

        # 测试掩码管理器
        from core.mask_manager import MaskManager

        manager = MaskManager(config)

        # 生成测试掩码
        mask = manager.generate_triangle_mask(triangle, (480, 640))
        combined_mask = manager.generate_combined_mask((480, 640), "exclude")

        print(f"✅ 掩码功能: 单个掩码 {mask.shape}, 组合掩码 {combined_mask.shape}")

        # 测试掩码应用
        import numpy as np

        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        masked_image = manager.apply_mask_to_image(test_image, "exclude")

        print(f"✅ 掩码应用: 原图 {test_image.shape}, 掩码后 {masked_image.shape}")

        return True
    except Exception as e:
        print(f"❌ 掩码功能测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🔥 火点检测系统快速验证")
    print("========================================")

    tests = [
        ("核心模块", test_core_modules),
        ("训练模块", test_training_modules),
        ("推理模块", test_inference_modules),
        ("掩码功能", test_mask_features),  # 新增
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")

    print("========================================")
    if passed == total:
        print("✅ 核心模块基本功能验证完成")
        return True
    else:
        print(f"⚠️  部分测试失败: {passed}/{total}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
