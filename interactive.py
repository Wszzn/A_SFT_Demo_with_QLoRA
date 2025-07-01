"""
交互式界面 - 猫娘的诞生
"""

import argparse
import logging
from generator import Generator
from config import GenerationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="猫娘对话模拟器 - 交互模式")
    parser.add_argument("--model_path", type=str, required=True,
                       help="微调模型路径")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B",
                       help="基础模型名称")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="top-p采样参数")
    
    args = parser.parse_args()
    
    # 配置生成参数
    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # 加载生成器
    try:
        generator = Generator(args.model_path, args.base_model)
        generator.interactive_mode()
    except Exception as e:
        logger.error(f"启动失败: {e}")


if __name__ == "__main__":
    main()