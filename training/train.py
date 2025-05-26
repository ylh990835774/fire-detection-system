#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练入口脚本
提供命令行接口进行火点检测模型训练
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到sys.path
sys.path.append(str(Path(__file__).parent.parent))

from core.config_manager import get_config_manager

from training.data_manager import DataManager
from training.model_trainer import ModelTrainer


def setup_logging(log_level="INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("training.log")],
    )


def create_data_config(images_dir, labels_dir, output_dir, args):
    """创建数据集配置"""
    print("创建数据集配置...")

    data_manager = DataManager(output_dir)

    config_file = data_manager.create_yolo_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        dataset_name=args.dataset_name,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        class_names=args.class_names.split(",") if args.class_names else None,
    )

    print(f"数据集配置文件已创建: {config_file}")
    return config_file


def main():
    parser = argparse.ArgumentParser(description="火点检测模型训练")

    # 数据相关参数
    parser.add_argument("--images-dir", type=str, required=True, help="图像文件目录")
    parser.add_argument("--labels-dir", type=str, required=True, help="标签文件目录")
    parser.add_argument("--data-config", type=str, help="现有的数据集配置文件（如果提供，将跳过数据集创建）")
    parser.add_argument("--dataset-name", type=str, default="fire_detection", help="数据集名称")
    parser.add_argument("--class-names", type=str, default="fire", help="类别名称，用逗号分隔")

    # 数据划分参数
    parser.add_argument("--train-split", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-split", type=float, default=0.1, help="测试集比例")

    # 模型参数
    parser.add_argument("--model", type=str, default="yolov8n", help="模型名称 (yolov8n/s/m/l/x)")
    parser.add_argument("--input-size", type=int, default=640, help="输入图像尺寸")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="学习率")
    parser.add_argument("--device", type=str, default="auto", help="训练设备 (auto/cpu/cuda)")
    parser.add_argument("--workers", type=int, default=8, help="数据加载器进程数")

    # 输出参数
    parser.add_argument("--output-dir", type=str, default="runs/train", help="输出目录")
    parser.add_argument("--save-period", type=int, default=10, help="模型保存周期")

    # 其他参数
    parser.add_argument("--resume", action="store_true", help="从上次训练恢复")
    parser.add_argument("--export-formats", type=str, nargs="+", default=["onnx"], help="导出模型格式")
    parser.add_argument("--validate", action="store_true", help="训练后进行验证")
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别"
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取配置管理器
        config_manager = get_config_manager()

        # 更新训练配置
        config_updates = {
            "model_name": args.model,
            "input_size": args.input_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "workers": args.workers,
            "save_dir": str(output_dir),
            "save_period": args.save_period,
            "class_names": args.class_names.split(",") if args.class_names else ["fire"],
            "num_classes": len(args.class_names.split(",")) if args.class_names else 1,
        }

        config_manager.update_config("training", config_updates)
        training_config = config_manager.get_training_config()

        # 处理数据集配置
        if args.data_config and Path(args.data_config).exists():
            data_config_file = args.data_config
            logger.info(f"使用现有数据集配置: {data_config_file}")
        else:
            # 创建新的数据集配置
            data_config_file = create_data_config(args.images_dir, args.labels_dir, str(output_dir / "datasets"), args)

        # 验证数据集
        data_manager = DataManager(str(output_dir / "datasets"))
        validation_result = data_manager.validate_dataset(data_config_file)

        if not validation_result["valid"]:
            logger.error("数据集验证失败:")
            for error in validation_result["errors"]:
                logger.error(f"  - {error}")
            return 1

        if validation_result["warnings"]:
            logger.warning("数据集验证警告:")
            for warning in validation_result["warnings"]:
                logger.warning(f"  - {warning}")

        # 创建训练器
        trainer = ModelTrainer(training_config)

        # 开始训练
        logger.info("开始训练模型...")
        training_info = trainer.train(data_config=data_config_file, output_dir=str(output_dir), resume=args.resume)

        logger.info("训练完成!")
        logger.info(f"训练时长: {training_info['training_time']:.2f}秒")
        logger.info(f"最佳模型: {training_info['best_model_path']}")
        logger.info(f"最终指标: {training_info['final_metrics']}")

        # 导出模型
        if args.export_formats:
            logger.info("导出模型...")
            exported_models = trainer.export_model(export_formats=args.export_formats)

            for format_name, model_path in exported_models.items():
                logger.info(f"{format_name.upper()} 模型: {model_path}")

        # 验证模型
        if args.validate:
            logger.info("验证模型...")
            validation_metrics = trainer.validate(data_config_file)
            logger.info(f"验证指标: {validation_metrics}")

        logger.info("所有任务完成!")
        return 0

    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
