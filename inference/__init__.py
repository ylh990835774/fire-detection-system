#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理模块
提供模型推理、检测和部署功能
"""

from .detector import BaseDetector
from .fire_detector import FireDetector
from .inference_engine import InferenceEngine

__all__ = ["BaseDetector", "FireDetector", "InferenceEngine"]
