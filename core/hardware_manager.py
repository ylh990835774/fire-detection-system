#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬件管理器
支持 Nvidia GPU、AMD GPU、算能 AI 芯片等多种硬件平台的检测和管理
"""

import os
import platform
import subprocess
import warnings
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import logging

# 尝试导入硬件相关库
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch 未安装，某些硬件检测功能将不可用")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil 未安装，系统信息检测功能将受限")

try:
    import py3nvml.py3nvml as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    """硬件类型枚举"""
    CPU = "cpu"
    NVIDIA_GPU = "nvidia_gpu"
    AMD_GPU = "amd_gpu"
    INTEL_GPU = "intel_gpu"
    SOPHON_AI = "sophon_ai"  # 算能 AI 芯片
    UNKNOWN = "unknown"


class InferenceBackend(Enum):
    """推理后端枚举"""
    PYTORCH = "pytorch"
    ONNX_CPU = "onnx_cpu"
    ONNX_CUDA = "onnx_cuda"
    ONNX_DIRECTML = "onnx_directml"
    ONNX_OPENVINO = "onnx_openvino"
    TENSORRT = "tensorrt"
    SOPHON_SDK = "sophon_sdk"
    UNKNOWN = "unknown"


@dataclass
class HardwareInfo:
    """硬件信息数据类"""
    hardware_type: HardwareType
    device_name: str
    device_id: Optional[int] = None
    memory_total: Optional[int] = None  # MB
    memory_available: Optional[int] = None  # MB
    driver_version: Optional[str] = None
    compute_capability: Optional[str] = None
    backend: Optional[InferenceBackend] = None
    is_available: bool = True
    additional_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}


class HardwareManager:
    """硬件管理器类"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._hardware_info: List[HardwareInfo] = []
        self._detected = False
        self._initialize()
    
    def _initialize(self):
        """初始化硬件管理器"""
        self.logger.info("初始化硬件管理器...")
        self.detect_hardware()
    
    def detect_hardware(self) -> List[HardwareInfo]:
        """检测所有可用硬件"""
        if self._detected:
            return self._hardware_info
        
        self.logger.info("开始检测硬件配置...")
        self._hardware_info = []
        
        # 检测 CPU
        self._detect_cpu()
        
        # 检测 GPU
        self._detect_nvidia_gpu()
        self._detect_amd_gpu()
        self._detect_intel_gpu()
        
        # 检测算能 AI 芯片
        self._detect_sophon_ai()
        
        self._detected = True
        self.logger.info(f"硬件检测完成，发现 {len(self._hardware_info)} 个可用设备")
        
        return self._hardware_info
    
    def _detect_cpu(self):
        """检测 CPU 信息"""
        try:
            cpu_name = platform.processor() or "Unknown CPU"
            cpu_count = os.cpu_count() or 1
            
            # 获取内存信息
            memory_total = None
            memory_available = None
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                memory_total = memory.total // (1024 * 1024)  # 转换为MB
                memory_available = memory.available // (1024 * 1024)
            
            cpu_info = HardwareInfo(
                hardware_type=HardwareType.CPU,
                device_name=f"{cpu_name} ({cpu_count} cores)",
                device_id=0,
                memory_total=memory_total,
                memory_available=memory_available,
                backend=InferenceBackend.ONNX_CPU,
                additional_info={
                    "cpu_count": cpu_count,
                    "architecture": platform.machine()
                }
            )
            
            self._hardware_info.append(cpu_info)
            self.logger.info(f"检测到 CPU: {cpu_info.device_name}")
            
        except Exception as e:
            self.logger.error(f"CPU 检测失败: {e}")
    
    def _detect_nvidia_gpu(self):
        """检测 NVIDIA GPU"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_total = props.total_memory // (1024 * 1024)  # MB
                    memory_available = memory_total  # 简化处理
                    
                    # 尝试获取更详细的信息
                    driver_version = None
                    if NVML_AVAILABLE:
                        try:
                            nvml.nvmlInit()
                            handle = nvml.nvmlDeviceGetHandleByIndex(i)
                            driver_version = nvml.nvmlSystemGetDriverVersion().decode('utf-8')
                            # 获取实际可用内存
                            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                            memory_available = mem_info.free // (1024 * 1024)
                        except Exception as e:
                            self.logger.warning(f"NVML 获取详细信息失败: {e}")
                    
                    gpu_info = HardwareInfo(
                        hardware_type=HardwareType.NVIDIA_GPU,
                        device_name=props.name,
                        device_id=i,
                        memory_total=memory_total,
                        memory_available=memory_available,
                        driver_version=driver_version,
                        compute_capability=f"{props.major}.{props.minor}",
                        backend=InferenceBackend.ONNX_CUDA,
                        additional_info={
                            "multi_processor_count": props.multi_processor_count,
                            "max_threads_per_multi_processor": props.max_threads_per_multi_processor
                        }
                    )
                    
                    self._hardware_info.append(gpu_info)
                    self.logger.info(f"检测到 NVIDIA GPU: {gpu_info.device_name}")
                    
        except Exception as e:
            self.logger.error(f"NVIDIA GPU 检测失败: {e}")
    
    def _detect_amd_gpu(self):
        """检测 AMD GPU"""
        try:
            # 检查是否有 AMD GPU 相关工具
            result = subprocess.run(['rocm-smi', '--showproductname'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines[1:], 0):  # 跳过标题行
                    if line.strip():
                        gpu_info = HardwareInfo(
                            hardware_type=HardwareType.AMD_GPU,
                            device_name=line.strip(),
                            device_id=i,
                            backend=InferenceBackend.ONNX_DIRECTML,
                            additional_info={"platform": "ROCm"}
                        )
                        self._hardware_info.append(gpu_info)
                        self.logger.info(f"检测到 AMD GPU: {gpu_info.device_name}")
                        
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            # 尝试通过其他方式检测
            try:
                if platform.system() == "Windows":
                    # Windows 下尝试使用 DirectML
                    gpu_info = HardwareInfo(
                        hardware_type=HardwareType.AMD_GPU,
                        device_name="AMD GPU (DirectML)",
                        device_id=0,
                        backend=InferenceBackend.ONNX_DIRECTML,
                        additional_info={"platform": "DirectML"}
                    )
                    self._hardware_info.append(gpu_info)
                    self.logger.info("检测到 AMD GPU (通过 DirectML)")
            except Exception as e:
                self.logger.debug(f"AMD GPU 检测失败: {e}")
    
    def _detect_intel_gpu(self):
        """检测 Intel GPU"""
        try:
            # 检查 OpenVINO 环境
            try:
                import openvino
                gpu_info = HardwareInfo(
                    hardware_type=HardwareType.INTEL_GPU,
                    device_name="Intel GPU (OpenVINO)",
                    device_id=0,
                    backend=InferenceBackend.ONNX_OPENVINO,
                    additional_info={"platform": "OpenVINO"}
                )
                self._hardware_info.append(gpu_info)
                self.logger.info("检测到 Intel GPU (通过 OpenVINO)")
            except ImportError:
                pass
                
        except Exception as e:
            self.logger.debug(f"Intel GPU 检测失败: {e}")
    
    def _detect_sophon_ai(self):
        """检测算能 AI 芯片"""
        try:
            # 检查算能 SDK
            if os.path.exists("/opt/sophon") or os.path.exists("/usr/local/sophon"):
                ai_info = HardwareInfo(
                    hardware_type=HardwareType.SOPHON_AI,
                    device_name="Sophon AI Chip",
                    device_id=0,
                    backend=InferenceBackend.SOPHON_SDK,
                    additional_info={"platform": "Sophon"}
                )
                self._hardware_info.append(ai_info)
                self.logger.info("检测到算能 AI 芯片")
                
        except Exception as e:
            self.logger.debug(f"算能 AI 芯片检测失败: {e}")
    
    def get_available_hardware(self) -> List[HardwareInfo]:
        """获取所有可用硬件信息"""
        return [hw for hw in self._hardware_info if hw.is_available]
    
    def get_hardware_by_type(self, hardware_type: HardwareType) -> List[HardwareInfo]:
        """根据硬件类型获取硬件信息"""
        return [hw for hw in self._hardware_info 
                if hw.hardware_type == hardware_type and hw.is_available]
    
    def get_best_hardware(self) -> Optional[HardwareInfo]:
        """获取最佳硬件（优先级：NVIDIA GPU > AMD GPU > Intel GPU > 算能 > CPU）"""
        priority_order = [
            HardwareType.NVIDIA_GPU,
            HardwareType.AMD_GPU,
            HardwareType.INTEL_GPU,
            HardwareType.SOPHON_AI,
            HardwareType.CPU
        ]
        
        for hw_type in priority_order:
            hardware_list = self.get_hardware_by_type(hw_type)
            if hardware_list:
                # 如果有多个同类型硬件，选择内存最大的
                return max(hardware_list, 
                          key=lambda x: x.memory_total or 0)
        
        return None
    
    def get_recommended_backend(self, hardware_info: HardwareInfo) -> InferenceBackend:
        """根据硬件信息推荐最佳推理后端"""
        if hardware_info.backend:
            return hardware_info.backend
        
        # 根据硬件类型推荐后端
        backend_mapping = {
            HardwareType.NVIDIA_GPU: InferenceBackend.ONNX_CUDA,
            HardwareType.AMD_GPU: InferenceBackend.ONNX_DIRECTML,
            HardwareType.INTEL_GPU: InferenceBackend.ONNX_OPENVINO,
            HardwareType.SOPHON_AI: InferenceBackend.SOPHON_SDK,
            HardwareType.CPU: InferenceBackend.ONNX_CPU,
        }
        
        return backend_mapping.get(hardware_info.hardware_type, InferenceBackend.ONNX_CPU)
    
    def print_hardware_summary(self):
        """打印硬件配置摘要"""
        print("\n" + "="*60)
        print("硬件配置摘要")
        print("="*60)
        
        if not self._hardware_info:
            print("未检测到可用硬件")
            return
        
        for i, hw in enumerate(self._hardware_info, 1):
            print(f"\n{i}. {hw.device_name}")
            print(f"   类型: {hw.hardware_type.value}")
            print(f"   设备ID: {hw.device_id}")
            
            if hw.memory_total:
                print(f"   总内存: {hw.memory_total:,} MB")
            if hw.memory_available:
                print(f"   可用内存: {hw.memory_available:,} MB")
            if hw.driver_version:
                print(f"   驱动版本: {hw.driver_version}")
            if hw.compute_capability:
                print(f"   计算能力: {hw.compute_capability}")
            
            print(f"   推荐后端: {self.get_recommended_backend(hw).value}")
            print(f"   状态: {'可用' if hw.is_available else '不可用'}")
        
        best_hw = self.get_best_hardware()
        if best_hw:
            print(f"\n推荐使用: {best_hw.device_name} ({best_hw.hardware_type.value})")
        
        print("="*60)


# 全局硬件管理器实例
hardware_manager = HardwareManager()


def get_hardware_manager() -> HardwareManager:
    """获取硬件管理器实例"""
    return hardware_manager


if __name__ == "__main__":
    # 测试代码
    hw_manager = HardwareManager()
    hw_manager.print_hardware_summary()