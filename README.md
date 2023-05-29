# TigerBot

<p align="center" width="100%">
<img src="http://x-pai.algolet.com/bot/img/logo_core.png" alt="Tiger" style="width: 50%; min-width: 300px; display: block; margin: auto;"></a>
</p>

## 最近更新

## 目录

- [安装](#安装)
- [模型](#模型)
- [推理](#模型推理)
- [推理](#量化)
- [测评](#测评)
- [预训练](#预训练)
- [微调](#微调)
- [API](#api)
- [TigerDoc](#tigerdoc)

## 安装

### 下载安装

1. 下载本仓库内容至本地/远程服务器

```bash
git clone https://github.com/TigerResearch/TigerBot
cd TigerBot
```

2. 创建conda环境

```bash
conda create --name tigerbot python=3.8
conda activate tigerbot
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

## 模型

- [tigerbot-7b-base](https://huggingface.co): Tigerbot基座模型，在高质量中英文语料上自监督预训练得到，预训练语料包含约700B单词
- [tigerbot-7b-sft](https://huggingface.co): 在约XXX万多轮对话数据上微调得到的
- [tigerbot-176b-base](https://huggingface.co)
- [tigerbot-176b-sft](https://huggingface.co)

## 推理

#### 单卡推理
```
CUDA_VISIBLE_DEVICES=0 python infer ${MODEL_DIR}
```

#### 多卡推理
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python infer ${MODEL_DIR}
```

## 量化
我们使用[GPTQ](https://github.com/IST-DASLab/gptq)算法和[GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa)实现量化：
```
# go to gptq path
cd gptq

# Save compressed model
CUDA_VISIBLE_DEVICES=0 python llama.py ${MODEL_DIR} c4 --wbits 4 --act-order --groupsize 128 --save tigerbot-4bit-128g.pt
```

#### 量化模型单卡推理
```
CUDA_VISIBLE_DEVICES=0 python infer ${MODEL_DIR} --wbits 4 --groupsize 128 --load --load tigerbot-4bit-128g.pt
```
#### 量化模型多卡推理
```
CUDA_VISIBLE_DEVICES=0,1 python infer ${MODEL_DIR} --wbits 4 --groupsize 128 --load --load tigerbot-4bit-128g.pt
```


## 测评