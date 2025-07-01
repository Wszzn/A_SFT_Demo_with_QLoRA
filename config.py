"""
配置文件 - 猫娘的诞生
"""

import torch
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "../../Qwen3-8B"
    trust_remote_code: bool = True
    device_map: str = "auto"
    
    # 量化配置
    use_quantization: bool = True
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16


@dataclass
class LoRAConfig:
    """LoRA配置"""
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    bias: str = "none"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class TrainingConfig:
    """训练配置"""
    output_dir: str = "./qwen-cat-finetune"
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 27
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    logging_dir: str = "./logs"
    logging_steps: int = 50
    save_strategy: str = "epoch"
    bf16: bool = True
    report_to: str = "none"
    max_data_size: int = 12000
    max_sequence_length: int = 64


@dataclass
class DataConfig:
    """数据配置"""
    corpus_file: str = "data.json"
    max_length: int = 64
    validation_split: float = 0.1
    test_split: float = 0.1


@dataclass
class GenerationConfig:
    """生成配置"""
    max_new_tokens: int = 88
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    early_stopping: bool = True


@dataclass
class Config:
    """主配置类"""
    model: ModelConfig = None
    lora: LoRAConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    generation: GenerationConfig = None
    
    # 全局设置
    random_seed: int = 20031009
    device: Optional[str] = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.lora is None:
            self.lora = LoRAConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.generation is None:
            self.generation = GenerationConfig()
            
        # 自动检测设备
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# 默认配置实例
default_config = Config()

# 快速配置预设
QUICK_CONFIGS = {
    "fast": Config(
        training=TrainingConfig(
            per_device_train_batch_size=8,
            num_train_epochs=1,
            max_data_size=5000
        )
    ),
    "quality": Config(
        training=TrainingConfig(
            per_device_train_batch_size=4,
            num_train_epochs=3,
            max_data_size=20000,
            learning_rate=1e-4
        )
    ),
    "gpu_friendly": Config(
        model=ModelConfig(use_quantization=True),
        training=TrainingConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8
        )
    )
}