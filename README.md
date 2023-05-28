# TigerBot

<p align="center" width="100%">
<img src="https://github.com/TigerResearch/TigerBot/blob/main/images/tiger.png" alt="Tiger" style="width: 50%; min-width: 300px; display: block; margin: auto;"></a>
</p>

## 目录

- [安装](#安装)
- [模型下载](#模型下载)
- [模型推理](#模型推理)
- [测评](#测评)
- [预训练](#预训练)
- [精调](#精调)
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

## 模型下载

## 模型推理

#### 模型列表

- [tiger-7b](https://huggingface.co)
- [tiger-176b](https://huggingface.co)

#### 单卡推理

```
CUDA_VISIBLE_DEVICES=0 python3 infer /path/to/model/weights
```

#### 多卡推理

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 infer /path/to/model/weights
```

#### 模型量化

```
cd gptq
CUDA_VISIBLE_DEVICES=0 python3 tigerbot.py /path/to/model/weights 
```

#### 量化模型推理

```
CUDA_VISIBLE_DEVICES=0 python3 infer --model-path /path/to/model/weights
```