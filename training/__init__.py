#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练模块
提供数据管理、模型训练等功能
"""

from .data_manager import DataManager
from .model_trainer import ModelTrainer

__all__ = ["DataManager", "ModelTrainer"]
