#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火点检测系统安装脚本
支持多硬件平台的 YOLO 目标检测系统
"""

from setuptools import setup, find_packages
import os

# 读取 README 文件
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "火点检测系统 - 基于YOLO的多硬件目标检测解决方案"

# 读取依赖列表
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过注释和空行
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

setup(
    name="fire-detection-system",
    version="1.0.0",
    author="AI Assistant",
    author_email="ai@example.com",
    description="基于YOLO的多硬件异常火点检测系统",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/fire-detection-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Graphics :: Viewers",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "gpu": [
            "onnxruntime-gpu>=1.16.0",
        ],
        "openvino": [
            "openvino>=2023.1.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "web": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "gradio>=3.40.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fire-train=training.train:main",
            "fire-detect=inference.deploy:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
        "configs": ["*.yaml", "*.yml"],
    },
    zip_safe=False,
    keywords=[
        "computer vision", 
        "object detection", 
        "YOLO", 
        "fire detection", 
        "deep learning",
        "pytorch",
        "onnx",
        "multi-hardware"
    ],
)