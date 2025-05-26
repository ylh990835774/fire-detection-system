#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火点检测系统快速验证脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def quick_test():
    """快速测试核心功能"""
    print("🔥 火点检测系统快速验证")
    print("=" * 40)

    # 1. 测试硬件管理器
    try:
        from core.hardware_manager import get_hardware_manager

        hw_manager = get_hardware_manager()
        devices = hw_manager.detect_hardware()
        print(f"✅ 硬件管理器: 检测到 {len(devices)} 个设备")
    except Exception as e:
        print(f"❌ 硬件管理器: {e}")

    # 2. 测试配置管理器
    try:
        from core.config_manager import get_config_manager

        config_manager = get_config_manager()
        training_config = config_manager.get_training_config()
        print(f"✅ 配置管理器: 训练模型 {training_config.model_name}")
    except Exception as e:
        print(f"❌ 配置管理器: {e}")

    # 3. 测试模型管理器
    try:
        from core.model_manager import get_model_manager

        model_manager = get_model_manager()
        stats = model_manager.get_model_statistics()
        print(f"✅ 模型管理器: {stats['total_models']} 个模型")
    except Exception as e:
        print(f"❌ 模型管理器: {e}")

    # 4. 测试数据管理器
    try:
        from training.data_manager import DataManager

        dm = DataManager("./test_data")
        print("✅ 数据管理器: 初始化成功")
    except Exception as e:
        print(f"❌ 数据管理器: {e}")

    # 5. 测试模型训练器
    try:
        from training.model_trainer import ModelTrainer

        trainer = ModelTrainer()
        device = trainer.setup_device()
        print(f"✅ 模型训练器: 设备 {device}")
    except Exception as e:
        print(f"❌ 模型训练器: {e}")

    # 6. 测试检测器基类
    try:
        from inference.detector import DetectorFactory

        detectors = DetectorFactory.list_detectors()
        print(f"✅ 检测器基类: {len(detectors)} 个注册检测器")
    except Exception as e:
        print(f"❌ 检测器基类: {e}")

    print("=" * 40)
    print("✅ 核心模块基本功能验证完成")


if __name__ == "__main__":
    quick_test()
