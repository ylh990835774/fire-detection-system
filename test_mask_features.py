#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三角形掩码功能测试脚本
验证新增的掩码功能是否正常工作
"""

import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_mask_config():
    """测试掩码配置功能"""
    print("=== 测试掩码配置 ===")

    try:
        from core.mask_config import MaskConfig, TriangularMask

        # 测试三角形掩码创建
        triangle = TriangularMask(
            vertices=[(100, 100), (200, 100), (150, 200)],
            name="test_triangle",
            description="测试三角形",
            mask_type="exclude",
        )

        print(f"✅ 三角形掩码创建成功: {triangle.name}")

        # 测试点包含判断
        assert triangle.contains_point(150, 150) == True  # 应该在内部
        assert triangle.contains_point(50, 50) == False  # 应该在外部
        print("✅ 点包含判断正确")

        # 测试坐标转换
        normalized = triangle.to_normalized_coordinates(640, 480)
        assert normalized.coordinate_type == "normalized"
        print("✅ 坐标转换正确")

        # 测试掩码配置
        config = MaskConfig()
        config.add_mask(triangle)
        assert len(config.triangular_masks) == 1
        print("✅ 配置管理正确")

        return True

    except Exception as e:
        print(f"❌ 掩码配置测试失败: {e}")
        return False


def test_mask_manager():
    """测试掩码管理器"""
    print("=== 测试掩码管理器 ===")

    try:
        from core.mask_config import MaskConfig, TriangularMask
        from core.mask_manager import MaskManager

        # 创建测试配置
        config = MaskConfig()
        triangle = TriangularMask(vertices=[(50, 50), (150, 50), (100, 150)], name="test_exclude", mask_type="exclude")
        config.add_mask(triangle)

        # 测试管理器
        manager = MaskManager(config)

        # 测试掩码生成
        mask = manager.generate_triangle_mask(triangle, (480, 640))
        assert mask.shape == (480, 640)
        assert mask.dtype == bool
        print("✅ 三角形掩码生成正确")

        # 测试组合掩码
        combined = manager.generate_combined_mask((480, 640), "exclude")
        assert combined.shape == (480, 640)
        print("✅ 组合掩码生成正确")

        # 测试图像掩码应用
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128  # 灰色图像
        masked_image = manager.apply_mask_to_image(test_image, "exclude", fill_value=0)
        assert masked_image.shape == test_image.shape
        print("✅ 图像掩码应用正确")

        # 测试检测结果过滤
        boxes = np.array([[75, 75, 125, 125], [200, 200, 250, 250]])  # 一个在掩码内，一个在外
        scores = np.array([0.8, 0.9])
        class_ids = np.array([0, 0])

        filtered_boxes, filtered_scores, filtered_class_ids = manager.filter_detections_by_mask(
            boxes, scores, class_ids, (480, 640), overlap_threshold=0.5
        )

        # 第一个检测框应该被过滤掉（在排除区域内）
        assert len(filtered_boxes) == 1
        print("✅ 检测结果过滤正确")

        return True

    except Exception as e:
        print(f"❌ 掩码管理器测试失败: {e}")
        return False


def test_mask_utils():
    """测试掩码工具函数"""
    print("=== 测试掩码工具函数 ===")

    try:
        from utils.mask_utils import (
            create_triangle_mask,
            point_in_triangle,
            triangle_area,
            triangle_centroid,
            validate_triangle,
        )

        triangle = [(100, 100), (200, 100), (150, 200)]

        # 测试点在三角形判断
        assert point_in_triangle(150, 150, triangle) == True
        assert point_in_triangle(50, 50, triangle) == False
        print("✅ 点在三角形判断正确")

        # 测试面积计算
        area = triangle_area(triangle)
        assert area > 0
        print(f"✅ 三角形面积计算正确: {area}")

        # 测试重心计算
        centroid = triangle_centroid(triangle)
        assert isinstance(centroid, tuple) and len(centroid) == 2
        print(f"✅ 重心计算正确: {centroid}")

        # 测试三角形验证
        assert validate_triangle(triangle) == True
        assert validate_triangle([(0, 0), (0, 0), (1, 1)]) == False  # 重复点
        print("✅ 三角形验证正确")

        # 测试掩码创建
        mask = create_triangle_mask((480, 640), triangle)
        assert mask.shape == (480, 640)
        print("✅ 掩码创建正确")

        return True

    except Exception as e:
        print(f"❌ 掩码工具函数测试失败: {e}")
        return False


def test_masked_detector():
    """测试支持掩码的检测器"""
    print("=== 测试支持掩码的检测器 ===")

    try:
        from inference.masked_detector import MaskedDetector

        # 创建一个模拟的掩码检测器
        class MockMaskedDetector(MaskedDetector):
            def load_model(self):
                self.is_loaded = True
                return True

            def _preprocess(self, image):
                return image

            def _inference(self, preprocessed_image):
                # 模拟返回一些检测结果
                return np.array([[[100, 100, 150, 150, 0.8, 0]]])  # [x1, y1, x2, y2, conf, class]

            def _postprocess(self, raw_output, image_shape):
                from inference.detector import DetectionResult

                boxes = raw_output[0][:, :4]
                scores = raw_output[0][:, 4]
                class_ids = raw_output[0][:, 5].astype(int)
                return DetectionResult(boxes, scores, class_ids, ["fire"], image_shape)

        # 创建测试检测器
        detector = MockMaskedDetector(model_path="dummy_path", class_names=["fire"])

        # 添加排除区域
        success = detector.add_exclude_triangle(vertices=[(50, 50), (200, 50), (125, 200)], name="test_exclude")
        assert success == True
        print("✅ 添加排除区域成功")

        # 测试掩码信息获取
        mask_info = detector.get_mask_info()
        assert mask_info["total_masks"] >= 1
        print("✅ 掩码信息获取正确")

        # 测试掩码设置验证
        validation = detector.validate_mask_setup()
        assert validation["mask_manager_loaded"] == True
        print("✅ 掩码设置验证正确")

        # 创建测试图像
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # 测试检测（不启用掩码）
        result1 = detector.detect(test_image, apply_mask=False)
        original_detections = result1.num_detections

        # 测试检测（启用掩码）
        result2 = detector.detect(test_image, apply_mask=True)
        masked_detections = result2.num_detections

        print(f"✅ 检测功能正常: 原始检测 {original_detections}, 掩码后检测 {masked_detections}")

        # 测试可视化功能
        result, vis_image = detector.detect_with_mask_visualization(test_image)
        assert vis_image.shape == test_image.shape
        print("✅ 掩码可视化正确")

        return True

    except Exception as e:
        print(f"❌ 支持掩码的检测器测试失败: {e}")
        return False


def test_config_file_loading():
    """测试配置文件加载"""
    print("=== 测试配置文件加载 ===")

    try:
        from core.mask_config import MaskConfig

        # 测试从文件加载配置
        config_path = Path(__file__).parent / "configs" / "mask_config.yaml"

        if config_path.exists():
            config = MaskConfig.load_from_file(config_path)
            print(f"✅ 配置文件加载成功，包含 {len(config.triangular_masks)} 个掩码")
        else:
            print("⚠️  配置文件不存在，跳过文件加载测试")

        # 测试配置保存
        test_config = MaskConfig()
        from core.mask_config import TriangularMask

        triangle = TriangularMask(vertices=[(10, 10), (50, 10), (30, 50)], name="test_save", mask_type="exclude")
        test_config.add_mask(triangle)

        test_save_path = Path(__file__).parent / "test_mask_config.yaml"

        # 确保清理之前的测试文件
        if test_save_path.exists():
            test_save_path.unlink()

        test_config.save_to_file(test_save_path)

        # 验证保存的文件
        if test_save_path.exists():
            try:
                loaded_config = MaskConfig.load_from_file(test_save_path)
                assert len(loaded_config.triangular_masks) == 1
                print("✅ 配置文件保存和加载正确")
            finally:
                # 确保清理测试文件
                if test_save_path.exists():
                    test_save_path.unlink()

        return True

    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        # 确保清理测试文件
        test_save_path = Path(__file__).parent / "test_mask_config.yaml"
        if test_save_path.exists():
            test_save_path.unlink()
        return False


def test_integration():
    """集成测试 - 测试完整的掩码工作流程"""
    print("=== 集成测试 ===")

    try:
        from core.mask_config import TriangularMask, get_mask_config
        from core.mask_manager import get_mask_manager

        # 获取全局管理器
        manager = get_mask_manager()
        config = get_mask_config()

        # 添加测试掩码
        test_triangle = TriangularMask(
            vertices=[(0.2, 0.2), (0.4, 0.2), (0.3, 0.4)],  # 归一化坐标
            name="integration_test",
            coordinate_type="normalized",
            mask_type="exclude",
        )

        config.add_mask(test_triangle)

        # 测试在不同图像尺寸下的工作
        for size in [(480, 640), (720, 1280), (1080, 1920)]:
            mask = manager.generate_combined_mask(size, "exclude")
            assert mask.shape == size

        print("✅ 多尺寸图像掩码生成正确")

        # 清理测试掩码
        config.remove_mask("integration_test")

        return True

    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False


def main():
    """运行所有掩码功能测试"""
    print("🔥 火点检测系统 - 三角形掩码功能测试")
    print("=" * 60)

    test_functions = [
        test_mask_config,
        test_mask_manager,
        test_mask_utils,
        test_masked_detector,
        test_config_file_loading,
        test_integration,
    ]

    passed_tests = 0
    total_tests = len(test_functions)

    for test_func in test_functions:
        try:
            if test_func():
                passed_tests += 1
            print()  # 空行分隔
        except Exception as e:
            print(f"❌ {test_func.__name__} 执行异常: {e}")
            print()

    print("=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)

    for i, test_func in enumerate(test_functions, 1):
        status = "✅ 通过" if i <= passed_tests else "❌ 失败"
        print(f"{test_func.__name__:<25} {status}")

    print("-" * 60)
    print(f"总计: {passed_tests}/{total_tests} 项测试通过")

    if passed_tests == total_tests:
        print("🎉 所有掩码功能测试通过！")
        return True
    else:
        print("⚠️  部分测试失败，请检查上述错误信息")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
