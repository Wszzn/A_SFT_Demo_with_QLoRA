# 🎭 生成器

基于Qwen大语言模型微调的猫娘的诞生系统，使用LoRA技术实现高效微调。

## ✨ 特性

- 🤖 基于Qwen3-8B模型微调
- 🔧 使用LoRA技术，参数高效微调
- 💾 支持4-bit量化，降低显存需求
- 🚀 快速推理，实时生成

## 🛠️ 环境要求

### 硬件要求
- GPU: 建议24GB+显存（支持CUDA）
- CPU: 多核处理器
- 内存: 8GB+

### 软件依赖
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU版本)

## 📦 安装

1. 克隆项目
```bash
git clone https://github.com/Wszzn/A SFT Demo with QLoRA.git
cd chinese-poem-generator
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 准备数据
将您的训练数据json文件放置为 `data.json`

## 🚀 快速开始

### 训练模型
```bash
python train.py
```


## 📊 模型架构

- **基础模型**: Qwen/Qwen3-8B
- **微调方法**: LoRA (Low-Rank Adaptation)
- **量化**: 4-bit NF4量化
- **目标模块**: q_proj, k_proj, v_proj, o_proj
- **LoRA秩**: 8
- **LoRA Alpha**: 16

## 📈 训练配置

```python
training_config = {
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "epochs": 27,
    "learning_rate": 2e-4,
    "scheduler": "cosine",
    "warmup_ratio": 0.1
}
```

## 📁 项目结构

```
A SFT Demo with QLoRA/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── train.py              # 主训练脚本
├── poem_generator.py     # 生成器类
├── interactive.py        # 交互式界面
├── dataset.py           # 数据集处理
├── config.py            # 配置文件
```

## 📖 数据格式
Json数据文件格式：
```
[
    {
        "instruction": "睡不着怎么办？",
        "output": "呜...主人也睡不着吗？喵~我刚刚在窗台上数星星呢，一只两只三只...可是数到一千只的时候发现窗外有只飞蛾在扑腾，好吵啊~\n\n要不要让主人抱着我睡觉呀？我的呼噜声可是有安眠曲的效果哦！或者我可以给主人唱首摇篮曲，虽然可能有点跑调...但是绝对很温柔的！\n\n*轻轻蹭了蹭主人的手*\n\n主人要不要和我一起数星星啊？我知道很多星座的故事呢！虽然有时候会分心想去追流星...但是一定会陪着主人直到睡着为止！"
    }, ...
]
```

## 🔧 配置说明

在 `config.py` 中可以调整：
- 模型参数
- 训练超参数  
- 数据处理配置
- 生成参数
- 其他配置

## 📝 使用说明

### 训练模型
运行 `train.py` 脚本，按照提示输入参数进行训练。

### 生成诗歌
运行 `interactive.py` 脚本，按照提示输入参数进行交互。


⭐ 如果这个项目对您有帮助，请给个Star支持一下！
