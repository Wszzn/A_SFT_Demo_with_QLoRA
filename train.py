"""
训练脚本 - 猫娘的诞生
"""

import os
import torch
import time
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType

from config import Config, default_config
from dataset import  load_corpus2, CatGirlDataset
from generator import Generator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PoemTrainer:
    """猫娘模型训练器"""
    
    def __init__(self, config: Config = None):
        self.config = config or default_config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # 设置随机种子
        torch.manual_seed(self.config.random_seed)
        
    def setup_environment(self):
        """环境设置和检查"""
        logger.info("🎭 猫娘的诞生 - Qwen3-8B微调实现")
        logger.info("=" * 55)
        
        logger.info("🖥️  正在检查运行环境...")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✅ GPU可用: {gpu_name}")
            logger.info(f"   显存容量: {gpu_memory:.1f} GB")
            logger.info(f"   CUDA版本: {torch.version.cuda}")
        else:
            logger.warning("⚠️  GPU不可用，将使用CPU运行（速度会很慢）")
        
        logger.info(f"🎯 使用设备: {self.config.device}")
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        logger.info("🤖 正在加载Qwen模型...")
        logger.info(f"📦 模型: {self.config.model.name}")
        
        # 量化配置
        bnb_config = None
        if self.config.model.use_quantization:
            logger.info("⚙️  配置4-bit量化以节省显存...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config.model.load_in_4bit,
                bnb_4bit_use_double_quant=self.config.model.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.model.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=self.config.model.bnb_4bit_compute_dtype
            )
        
        # 加载分词器
        logger.info("🔤 加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name, 
            trust_remote_code=self.config.model.trust_remote_code
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"✅ 分词器词汇表大小: {len(self.tokenizer):,}")
        
        # 加载模型
        logger.info("🧠 加载预训练模型...")
        start_time = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            quantization_config=bnb_config,
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code
        )
        load_time = time.time() - start_time
        logger.info(f"✅ 模型加载完成 (耗时: {load_time:.2f}秒)")
    
    def setup_lora(self):
        """设置LoRA配置"""
        logger.info("🔧 配置LoRA微调参数...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            target_modules=self.config.lora.target_modules,
            bias=self.config.lora.bias,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # 统计参数
        trainable_params = 0
        total_params = 0
        for param in self.model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info("📊 可训练参数统计:")
        logger.info(f"   📈 可训练参数: {trainable_params:,}")
        logger.info(f"   📊 总参数量: {total_params:,}")
        logger.info(f"   🎯 训练参数占比: {100 * trainable_params / total_params:.3f}%")
    
    def prepare_dataset(self):
        """准备训练数据"""
        logger.info("📚 正在加载猫娘语录...")
        
        # 加载语料
        corpus = load_corpus2(self.config.data.corpus_file)
        
        # 限制数据大小
        max_size = self.config.training.max_data_size
        actual_size = min(len(corpus), max_size)
        logger.info(f"📊 使用数据量: {actual_size:,} / {len(corpus):,}")
        
        # 创建数据集
        logger.info("🎯 准备训练数据...")
        self.dataset = CatGirlDataset(
            corpus[:actual_size], 
            self.tokenizer,
            max_len=self.config.training.max_sequence_length
        )
        
        if len(self.dataset) == 0:
            raise ValueError("没有有效的训练数据")
        
        logger.info(f"✅ 训练集准备完成: {len(self.dataset):,} 个猫娘对话样本")
        
        # 显示样本
        logger.info("📝 训练样本预览:")
        for i, sample in enumerate(self.dataset.get_sample_dialogues(3)):
            logger.info(f"   {i+1}. 我: {sample['instruction']}")
            logger.info(f"      猫娘: {sample['output']}")
    
    def setup_trainer(self):
        """设置训练器"""
        logger.info("⚙️  训练配置:")
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            num_train_epochs=self.config.training.num_train_epochs,
            learning_rate=self.config.training.learning_rate,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            warmup_ratio=self.config.training.warmup_ratio,
            logging_dir=self.config.training.logging_dir,
            logging_steps=self.config.training.logging_steps,
            save_strategy=self.config.training.save_strategy,
            bf16=self.config.training.bf16,
            report_to=self.config.training.report_to,
        )
        
        logger.info(f"   📊 批次大小: {training_args.per_device_train_batch_size}")
        logger.info(f"   🔄 梯度累积步数: {training_args.gradient_accumulation_steps}")
        logger.info(f"   🎯 训练轮数: {training_args.num_train_epochs}")
        logger.info(f"   📈 学习率: {training_args.learning_rate}")
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True)
        
        # 初始化Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=data_collator,
        )
        
        # 计算训练步数
        total_steps = len(self.dataset) // (
            training_args.per_device_train_batch_size * 
            training_args.gradient_accumulation_steps
        )
        logger.info(f"📊 预计训练步数: {total_steps}")
    
    def train(self):
        """开始训练"""
        logger.info("🚀 开始微调训练...")
        logger.info("=" * 50)
        
        training_start = time.time()
        
        try:
            self.trainer.train()
            training_time = time.time() - training_start
            logger.info(f"🎉 训练完成! 总耗时: {training_time:.2f}秒 ({training_time/60:.1f}分钟)")
            return training_time
        except Exception as e:
            logger.error(f"❌ 训练过程中出现错误: {str(e)}")
            raise
    
    def save_model(self, save_path: str = None):
        """保存模型"""
        if save_path is None:
            save_path = "./qwen-cat-final"
        
        logger.info(f"💾 保存微调模型到: {save_path}")
        self.trainer.save_model(save_path)
        logger.info("✅ 模型保存完成")
        return save_path
    
    def evaluate_model(self, save_path: str):
        """评估模型"""
        logger.info("🎭 测试微调后的模型")
        logger.info("=" * 40)
        
        # 加载生成器
        generator = Generator(save_path, self.config.model.name)
        
        # 测试样本
        test_lines = [
            "你是谁？",
        ]
        
        logger.info("🎨 AI猫娘展示:")
        logger.info("-" * 30)
        
        results = generator.evaluate_samples(test_lines)
        
        for i, result in enumerate(results["results"], 1):
            logger.info(f"📝 测试 {i}/{len(test_lines)}:")
            logger.info(f"✨ 我: {result['input']}")
            logger.info(f"🎯 猫娘: {result['output']}")
            logger.info(f"⏱️  生成用时: {result['time']:.2f}秒")
            logger.info("─" * 25)
        
        return results
    
    def run_full_training(self):
        """完整训练流程"""
        try:
            # 1. 环境设置
            self.setup_environment()
            
            # 2. 加载模型
            self.load_model_and_tokenizer()
            
            # 3. 设置LoRA
            self.setup_lora()
            
            # 4. 准备数据
            self.prepare_dataset()
            
            # 5. 设置训练器
            self.setup_trainer()
            
            # 6. 训练
            training_time = self.train()
            
            # 7. 保存模型
            save_path = self.save_model()
            
            # 8. 评估模型
            eval_results = self.evaluate_model(save_path)
            
            # 9. 训练总结
            logger.info("🎊 程序执行完成!")
            logger.info("📊 训练总结:")
            logger.info(f"   🎯 训练数据: {len(self.dataset):,} 个对联")
            logger.info(f"   ⏱️  训练时长: {training_time:.2f}秒")
            logger.info(f"   💾 模型保存: {save_path}")
            logger.info(f"   🎭 测试样本: {len(eval_results['results'])} 个")
            logger.info("✨ 猫娘诞生! ✨")
            
            return {
                "model_path": save_path,
                "training_time": training_time,
                "dataset_size": len(self.dataset),
                "eval_results": eval_results
            }
            
        except Exception as e:
            logger.error(f"❌ 训练失败: {str(e)}")
            raise


def main():
    """主函数"""
    trainer = PoemTrainer()
    results = trainer.run_full_training()
    return results


if __name__ == "__main__":
    main()