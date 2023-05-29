# TigerBot

<p align="center" width="100%">
<img src="http://x-pai.algolet.com/bot/img/logo_core.png" alt="Tiger" style="width: 50%; min-width: 300px; display: block; margin: auto;"></a>
</p>

## 最近更新

## 目录

- [安装](#安装)
- [模型](#模型)
- [训练和推理](#训练和推理)
- [量化](#量化)
- [测评](#测评)
- [API](#API)

## 安装

### 下载安装

```bash

conda create --name tigerbot python=3.8
conda activate tigerbot
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

git clone https://github.com/TigerResearch/TigerBot
cd TigerBot
pip install -r requirements.txt
```

## 模型

<summary>Tigerbot-7B</summary>

| Tigerbot-7B                                    | Bits | memory(MiB) | 
|------------------------------------------------|------|-------------|
| [Tigerbot-7B-base](https://huggingface.co)     | 16   | -           |
| [Tigerbot-7B-sft](https://huggingface.co)      | 16   | -           |
| [Tigerbot-7B-sft-int4](https://huggingface.co) | 4    | -           |

<summary>Tigerbot-176B)</summary>

| Tigerbot-176B                                    | Bits | memory(MiB) |
|--------------------------------------------------|------|-------------|
| [Tigerbot-176B-sft](https://huggingface.co)      | 16   | 13940       |
| [Tigerbot-176B-sft-int4](https://huggingface.co) | 4    | -           |

## 训练和推理

### 预训练

启动训练前安装DeepSpeed

```
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
```

TORCH_CUDA_ARCH_LIST根据你运行的GPU架构做调整，获取TORCH_CUDA_ARCH_LIST

```
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"
```

如果返回的结果是(8, 0)，那么TORCH_CUDA_ARCH_LIST="8.0"

#### 训练数据

Tigerbot-7B-base在Bloom-7B初始化基础上进行预训练，训练数据包括：

- 中英自然语言文本
    - [中文书籍](https://huggingface.co)
    - [中文新闻](https://huggingface.co)
    - [中文百科](https://huggingface.co)
    - [英文书籍](https://huggingface.co)
    - [英文web文本](https://huggingface.co)
    - [英文百科](https://huggingface.co)
- 15种编程语言

#### 启动训练

```
deepspeed --include="localhost:0,1,2,3" train/train_clm.py \
--deepspeed ./ds_config/ds_config_zero3.json \
--model_path ${MODEL_DIR} \
--do_train \
--train_file_path ./data/dev_pretrain.json \
--output_dir ./ckpt-clm \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 5 \
--learning_rate  1.2e-4 \
--evaluation_strategy steps \
--eval_steps 10 \
--bf16 True \
--save_strategy steps \
--save_steps 10 \
--save_total_limit 2 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2
```

### 微调

#### 训练数据

#### 启动训练

```
deepspeed include="localhost:0,1,2,3" train/train_sft.py \
--deepspeed ./ds_config/ds_config_zero3.json \
--model_path ./tigerbot_560m \
--do_train \
--train_file_path ./data/dev_sft.json \
--output_dir ./ckpt-sft \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 5 \
--learning_rate 1e-5 \
--evaluation_strategy steps \
--eval_steps 10 \
--bf16 True \
--save_strategy steps \
--save_steps 10 \
--save_total_limit 2 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2
```

### 推理

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
cd gptq

# Save compressed model
CUDA_VISIBLE_DEVICES=0 python llama.py ${MODEL_DIR} c4 --wbits 4 --act-order --groupsize 128 --save tigerbot-4bit-128g.pt
```

#### 量化模型单卡推理

```
CUDA_VISIBLE_DEVICES=0 python infer ${MODEL_DIR} --wbits 4 --groupsize 128 --load tigerbot-4bit-128g.pt
```

#### 量化模型多卡推理

```
CUDA_VISIBLE_DEVICES=0,1 python infer ${MODEL_DIR} --wbits 4 --groupsize 128 --load tigerbot-4bit-128g.pt
```

## 测评

#### 英文自动化测评

英文自动化测评在7大传统NLP任务上进行，各模型细分得分情况如下：

![image](image/英文任务得分详细表.png)

## API

#### 对话（Chat-API）

#### 插件（Plug-ins）

#### 微调（Fine-Tunes）




