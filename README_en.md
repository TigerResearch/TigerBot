# TigerBot

<p align="center" width="100%">
<img src="image/tiger.jpg" alt="Tiger" style="width: 20%; display: block; margin: auto;"></a>
</p>
<p align="center">
<font face="黑体" size=5"> A cutting-edge foundation for your very own LLM. </font>
</p>
<p align="center">
   🌐 <a href="https://tigerbot.com/" target="_blank">TigerBot</a> • 🤗 <a href="https://huggingface.co/TigerResearch" target="_blank">Hugging Face</a>
</p>
<h4 align="left">
    <p>
        <b>English</b> |
        <a href="https://github.com/TigerResearch/TigerBot/blob/main/README.md">Chinese</a>
    <p>
</h4>

## News

- [7/08/2023] TigerBot 2023.07 (V2) release :fire:
   - We introduce tigerbot-7b-base (v2) trained on 1.5TB high quality data. The training was conducted on 1,000 gpus, took about 4 weeks and cost 3,000,000 Yuan (RMB). The evaluations based on public nlp Chinese and English datasets show that it outperforms bloom and llama with the same model size by 15-30%.
   - We introduce tigerbot-7b-sft (v2) which was built by finetuning tigerbot-7b-base (v2) with 20G high quality instruction data. It outperforms tigerbot-7b-sft-v1 by 9.3% on nine public datasets evaluation.
   
   - How to Use：
    ```python
    import transformers
    
    # If you have the v1 version in your cache, set `force_download=True`.
    model_sft = transformers.AutoModelForCausalLM.from_pretrained('TigerResearch/tigerbot-7b-sft', force_download=True)
    model_base = transformers.AutoModelForCausalLM.from_pretrained('TigerResearch/tigerbot-7b-base', force_download=True)
    ```
  - We are hosting internet plugin which enables web browsing with tigerbot. Tigerbot utilizes some mainstream search engines and some web tools (like weather, stock, calculator) to navigate results and interact with websites. Meanwhile , you can use tigerbot chat-api with internet search switch. [[TigerBot with search mode (default off) :earth_asia:](https://www.tigerbot.com/chat)][[paper](https://github.com/TigerResearch/TigerBot/wiki/TigerBot-upgraded-with-internet-search)]
  - You can use tigerbot chat-api with streaming switch [[TigerBot](https://www.tigerbot.com/chat)][[TigerBot-API](https://www.tigerbot.com/api-reference/chat)]
  - New features in tigerbot-api, including LLM (chat, plugin, finetune), text (embedding, summarization, pdf2text), vision (text2image) [[TigerBot-API](https://www.tigerbot.com/api-reference/chat)]
  
- [6/27/2023] PEFT TigerBot with QLoRA:  finetune a tigerbot-7b-sft model on single RTX3090 with qlora, speeds up by 16 times and reduces GPI3/4, which also preventing overfitting on downstream data[[code](https://github.com/TigerResearch/TigerBot/blob/main/train/train_with_qlora.py)] [[paper](https://github.com/TigerResearch/TigerBot/wiki/PEFT-TigerBot-7b-with-QLoRA,-building-an-domain-LLM-on-one-consumer-level-GPU-in-hours)] [[model](https://huggingface.co/TigerResearch/medical-bot-peft-from-tigerbot-7b-sft)]

<p align="center" width="100%">
	<img src="image/peft_metrics.png" alt="tigerbot chat-api sample" style="width: 65%; display: block; margin: auto;"></a>
</p>

- [6/26/2023] TigerBot now is on desktop! [Make your own chatbot with tigerbot and Svelte](#Community)，thanks to @SaraiQX ！
- [6/20/2023] How to use tigerbot api in langchian(<a href="https://github.com/TigerResearch/TigerBot/blob/main/apps/tigerbot_chatapi.py">sample code</a>) thansk to @wordweb ！

<p align="center" width="100%">
	<img src="image/tigerbot_chatapi_sample.png" alt="tigerbot chat-api sample" style="width: 65%; display: block; margin: auto;"></a>
</p>

- [6/13/2023] Plug-in api upgrades：[search results、prompt prefix and tf-idf, embedding mixture weights](#API)
- [6/13/2023] Fast way to [download model](#Model Weights)
- [6/13/2023] TigerBot now is on QQ! [QQ bot based on Tigerbot with custom knowledge base](#Community)，thanks to @wordweb ！
- [6/09/2023] Stream infer and web demo，thanks to @Tlntin ！
- [6/08/2023] Run tigerBot on [colab, windows, langchain and webui](#Community), thanks to @wordweb @runfuture !

## Abstract

TigerBot is a multi-language and multitask LLM. We evaluated our MVP model on public NLP datasets and found that our
model reached 96% of performance of OpenAI InstructGPT at the same model size. We hereby open-source our explorations as following:

- Model：TigerBot-7B, TigerBot-7B-base，TigerBot-180B (research version),
- Code:
    1. The whole training process codes including model pretraining and supervised fine-tuning.
    2. Model quantization with GPTQ.
    3. Inference on single GPU or multiple GPUs.
- Data:
    1. Pre-training data: 100GB pretraining data deduplicated and filtered low quality content from 2TB corpus.
    2. SFT data: 1GB (millions of) textual instructions. This dataset consists of 10 major user-instruction categories and 120 subcategories.
    3. Domain-specific data: We provide data into different domains: finance, law, and wikipedia.
- API: We provide APIs including chat, plugin, and finetune which allow users to create their own models and applications easily.

We pretrained and supervised fine-tuned our models, starting from a vanilla BLOOM, and made some algorithmic innovations so far:

- A stronger yet more elegant supervised learning algorithms to achieve higher learnability in supervised fine-tuning.
- We implemented a probabilistic modeling and ensemble approach to achieve better factuality and generativeness.
- We improved the memory management and multi-node communication of distributed training with deepspeed. It guarantees months of training in a thousand-gpu enviroment with zero downtime.
- We used a specialized tokenizer and supervised training algorithm better suited for otherwise more skewed Chinese language distribution.


## Contents

- [Install](#Install)
- [Model Weights](#Model-Weights)
- [Training and Inference](#Training-and-Inference)
- [Datasets](#Datasets)
- [Evaluation](#Evaluation)
- [API](#API)
- [Others](#Others)

## Install

```bash

conda create --name tigerbot python=3.8
conda activate tigerbot
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

git clone https://github.com/TigerResearch/TigerBot
cd TigerBot
pip install -r requirements.txt
```

## Model Weights

<summary>Tigerbot-7B</summary>

| Tigerbot-7B                                                                                 | Bits | memory(GB) |
| ------------------------------------------------------------------------------------------- | ---- | ---------- |
| [tigerbot-7b-base](https://huggingface.co/TigerResearch/tigerbot-7b-base)                   | 16   | 17.2       |
| [tigerbot-7b-sft](https://huggingface.co/TigerResearch/tigerbot-7b-sft)                     | 16   | 17.2       |
| [tigerbot-7b-sft-4bit-128g](https://huggingface.co/TigerResearch/tigerbot-7b-sft-4bit-128g) | 4    | 8.5        |

<summary>Tigerbot-180B-Research</summary>

| Tigerbot-180B-Research                                                                             | Bits | memory(GB) |
| -------------------------------------------------------------------------------------------------- | ---- | ---------- |
| [tigerbot-180b-sft](https://huggingface.co/TigerResearch/tigerbot-180b-research)                   | 16   | 347.6      |
| [tigerbot-180b-sft-4bit-128g](https://huggingface.co/TigerResearch/tigerbot-180b-research-4bit-128g) | 4    | 108.5      |

<details> 
<summary><b>versions</b></summary>

- tigerbot-7b-sft

  - tigerbot-7b-sft-v2 (2023.07.08) [[huggingface](https://huggingface.co/TigerResearch/tigerbot-7b-sft-v2)]

  - tigerbot-7b-sft-v1 (2023.06.07) [[huggingface](https://huggingface.co/TigerResearch/tigerbot-7b-sft-v1)]

- tigerbot-7b-base

  - tigerbot-7b-base-v2 (2023.07.08) [[huggingface](https://huggingface.co/TigerResearch/tigerbot-7b-base-v2)]
  - Tigerbot-7b-base-v1 (2023.06.07) [[huggingface](https://huggingface.co/TigerResearch/tigerbot-7b-base-v1)]

</details>

## Training and Inference

### Pre-training

Install DeepSpeed

```
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
```

Edit TORCH_CUDA_ARCH_LIST to insert the code for the architectures of the GPU cards you intend to use.

```
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"
```

So if you get 8, 0, then use TORCH_CUDA_ARCH_LIST="8.0".

command to start training

```
deepspeed \
--include="localhost:0,1,2,3" \
./train_clm.py \
--deepspeed ./ds_config/ds_config_zero3.json \
--model_name_or_path TigerResearch/tigerbot-7b-base \
--dataset_name TigerResearch/dev_pretrain \
--do_train \
--output_dir ./ckpt-clm \
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

### Fine-tunes

command to start training

```
deepspeed \
--include="localhost:0,1,2,3" \
./train_sft.py \
--deepspeed ./ds_config/ds_config_zero3.json \
--model_name_or_path TigerResearch/tigerbot-7b-base \
--dataset_name TigerResearch/dev_sft \
--do_train \
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

### Inference

You can infer with command line. Input `clear` to clean history and input `exit` to stop it.

<p width="100%">
    <img src="image/terminal_case.jpeg" alt="命令行推理" style="width: 100%; min-width: 200px;">
</p>

#### Infer with single GPU

`tigerbot-7b-sft` can be loaded for inference on RTX3090 GPU
```
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path ${MODEL_DIR}
```

If you want to enable streaming output, please replace `infer.py` with `infer_stream.py`, and the output will change from one-shot output to sentence-by-sentence output.
```
CUDA_VISIBLE_DEVICES=0 python ./other_infer/infer_stream.py --model_path ${MODEL_DIR}
```

If you want to enable the web interface for Q&A, change the model path corresponding to model_path on line 12 of `web_demo.py` to the path where your model is located, and then run the following command to enable the web interface.
```
CUDA_VISIBLE_DEVICES=0 python ./apps/web_demo.py
```

`tigerbot-7b-base` uses continuation (non-question answering) inference code.

```
CUDA_VISIBLE_DEVICES=0 python ./other_infer/infer_pretrain.py --model_path ${PRETRAIN_MODEL_DIR}
```

#### Infer with multiple GPUS

`tigerbot-180b-sft` can be loaded for parallelism inference on 5 A100(80G) GPUs

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python infer.py --model_path ${MODEL_DIR}
```


#### Deploy API
If you want to enable api, you need to install fastapi first, change the model path on line 193 to yours, and then run the service.
```bash
pip install "fastapi[all]"
python api.py
```

Then you can test the client to call the api through the web service
```bash
python ./apps/client.py
```

The client can also call the API asynchronously through the web service
```bash
python ./apps/async_client.py
```

It is also possible to call the web service to generate text through the previous web page.
```bash
python ./apps/web_api_demo.py
```

### Quantization

We use [GPTQ](https://github.com/IST-DASLab/gptq) and [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) to
quantize models.

go to the path of gptq

```
cd gptq
```

#### Model quantization

```
CUDA_VISIBLE_DEVICES=0 python tigerbot.py ${MODEL_DIR} c4 --wbits 4 --act-order --groupsize 128 --save ${MODEL_DIR}/tigerbot-7b-4bit-128g.pt
```

#### Quantized model infer with single GPU

[`tigerbot-7b-sft-4bit-128g`](https://huggingface.co/TigerResearch/tigerbot-7b-sft-4bit-128g) can be loaded for
inference on RTX3090 GPU

```
CUDA_VISIBLE_DEVICES=0 python tigerbot_infer.py ${MODEL_DIR} --wbits 4 --groupsize 128 --load ${MODEL_DIR}/tigerbot-7b-4bit-128g.pt
```

[`tigerbot-180b-research-4bit-128g`](https://huggingface.co/TigerResearch/tigerbot-180b-research-4bit-128g) can be
loaded for parallelism inference on 2 A100(80G) GPUs

```
CUDA_VISIBLE_DEVICES=0,1 python tigerbot_infer.py ${MODEL_DIR} --wbits 4 --groupsize 128 --load {MODEL_DIR}/tigerbot-4bit-128g.pt
```

For quantized model shards

```
CUDA_VISIBLE_DEVICES=0,1 python tigerbot_infer.py ${MODEL_DIR} --wbits 4 --groupsize 128 --load "{MODEL_DIR}/tigerbot-4bit-128g-*.pt"
```

## Datasets

### Pretraining Datasets

- <a href=https://huggingface.co/datasets/TigerResearch/pretrain_zh>Chinese Pretraining Corpus - 55G [hugging face]</a>
- <a href=https://huggingface.co/datasets/TigerResearch/pretrain_en>English Pretraining Corpus - 51G [hugging face]</a>

  | Type | Disk | Source |
  | ---------- | -------- | ---- |
  | zh-book | 12G | TigerBot |
  | zh-webtext | 25G | TigerBot |
  | zh-baike | 19G | TigerBot |
  | en-book | 22G | Public |
  | en-web | 6.9G | Public |
  | en-wiki | 22G | Public |
  | **Total**     | **106G** | |

- Distribution of Pre-training Data

<p align="center" width="100%">
<img src="image/pretrain_v2.png" alt="Tiger" style="width: 65%; display: block; margin: auto;"></a>
</p>

- Distribution of zh-book and coding data.

<p width="100%">
    <img src="image/zh-books.png" alt="中文书籍分类" style="width: 50%; min-width: 200px;"><img src="image/code-lang-type.png" alt="代码语言" style="width: 50%; min-width: 200px;">
</p>

### Supervised Fine-tuning Datasets

#### Data collection
- We collect SFT data as follows:
  a. self-instruct
  b. human-labeling
  c. open-source data cleaning

#### Data cleaning
We clean and filter data as follows:
- rule-based and keyword-based ways to filter low quality and unsafe contents.
- deduplicate

#### Datasets to open source
- 1200K Instruction-following dataset (download it from huggingface)

  | Type         | Language | Dataset                                                                                                                           | Number | Source |
  |--| ---- |--------|----| ------ |
  | alpaca-zh  | zh | [tigerbot-alpaca-zh-0.5m](https://huggingface.co/datasets/TigerResearch/tigerbot-alpaca-zh-0.5m)                                 | 500K   | TigerBot |
  | wiki-qa     | zh | [tigerbot-wiki-qa-1k](https://huggingface.co/datasets/TigerResearch/tigerbot-wiki-qa-zh-1k)                                      | 1K     | TigerBot |
  | book-qa     | zh | [tigerbot-book-qa-1k](https://huggingface.co/datasets/TigerResearch/tigerbot-book-qa-1k)                                         | 1K     | TigerBot |
  | riddle-qa       | zh | [tigerbot-riddle-qa-1k](https://huggingface.co/datasets/TigerResearch/tigerbot-riddle-qa-1k)                                     | 1K     | TigerBot |
  | mrc     | zh | [tigerbot-superclue-c3-zh-5k](https://huggingface.co/datasets/TigerResearch/tigerbot-superclue-c3-zh-5k)                         | 5K     | TigerBot |
  | HC3-qa         | zh | [tigerbot-HC3-zh-12k](https://huggingface.co/datasets/TigerResearch/tigerbot-HC3-zh-12k)                                         | 12K    | Public |
  | zhihu-qa     | zh | [tigerbot-zhihu-zh-10k](https://huggingface.co/datasets/TigerResearch/tigerbot-zhihu-zh-10k)                                     | 10K    | Public   |
  | alpaca-en  | en | [tigerbot-alpaca-en-50k](https://huggingface.co/datasets/TigerResearch/tigerbot-alpaca-en-50k)                                   | 50K    | TigerBot |
  | brainstorm     | en | [tigerbot-dolly-Brainstorming-en-1.7k](https://huggingface.co/datasets/TigerResearch/tigerbot-dolly-Brainstorming-en-1.7k)       | 1.7K   | Public   |
  | classify         | en | [tigerbot-dolly-Classification-en-2k](https://huggingface.co/datasets/TigerResearch/tigerbot-dolly-Classification-en-2k)         | 2K     | Public |
  | math     | en | [tigerbot-gsm-8k-en](https://huggingface.co/datasets/TigerResearch/tigerbot-gsm-8k-en)                                           | 8K     | Public |
  | code         | en | [tigerbot-kaggle-leetcodesolutions-en-2k](https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k) | 2K     | TigerBot |
  | recipe     | en | [tigerbot-kaggle-recipes-en-2k](https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-recipes-en-2k)                     | 2K     | Public |
  | medical-note     | en | [tigerbot-mt-note-generation-en](https://huggingface.co/datasets/TigerResearch/tigerbot-mt-note-generation-en)                   | 0.45K  | Public |
  | multi-run   | en | [tigerbot-OIG-multichat-en-50k](https://huggingface.co/datasets/TigerResearch/tigerbot-OIG-multichat-en-50k)                     | 50K    | TigerBot |
  | general     | en | [tigerbot-stackexchange-qa-en-0.5m](https://huggingface.co/datasets/TigerResearch/tigerbot-stackexchange-qa-en-0.5m)             | 500K    | Public |
  | wiki-qa    | en | [tigerbot-wiki-qa-bart-en-10k](https://huggingface.co/datasets/TigerResearch/tigerbot-wiki-qa-bart-en-10k)                       | 10K     | Public |
  | youtube-howto | en | [tigerbot-youtube-howto-en-50k](https://huggingface.co/datasets/TigerResearch/tigerbot-youtube-howto-en-50k)                     | 50K     | Public |
  | **Total**     |  |                                                                                                                                  | **1200K** |


### Domain-specific Data
- Domain-specific Data for Plugins

  | Type                                                                                       | Number        |
  |-----------------------------------------------------------------------------------------|-------------------|
  | [Finance-Research](https://huggingface.co/datasets/TigerResearch/tigerbot-research-plugin) | 5K       |
  | [Finance-Earning](https://huggingface.co/datasets/TigerResearch/tigerbot-earning-plugin)         | 1K       |
  | [Law](https://huggingface.co/datasets/TigerResearch/tigerbot-law-plugin)                    | 550K |
  | [Wiki](https://huggingface.co/datasets/TigerResearch/tigerbot-wiki-plugin)                   | 100K     |

## Evaluation

Evaluation result of V2 version SFT model

![image](image/evaluation_sft_v2.jpg)

Evaluation result of V2 version base model

![image](image/evaluation_base_v2.jpg)

<details>
Evaluation result of V1 version SFT and and base model

We evaluate our SFT models on seven public NLP datasets, and compare these with OpenAI-InstructGPT. 
Results against OpenAI-InstructGPT-6B-SFT.

![image](image/auto-valuation-1.png)

We evaluate our Pretrained models on seven public NLP datasets.
Results against bloom-7b1.

![image](image/auto-valuation-2.png)

</details>

## API

<details>

### [chat](https://www.tigerbot.com/api-reference/chat)

<details><summary><b>Example</b></summary>
<img src="image/api/demo/chat.png" alt="tigerbot chat-api sample" style="width: 65%; display: block">
<img src="image/api/demo/chat2.png" alt="tigerbot chat-api sample" style="width: 65%; display: block">
</details>

### [plugin](https://www.tigerbot.com/api-reference/plugin)

<details><summary><b>Example</b></summary>
<img src="image/api/demo/plugin.png" alt="tigerbot chat-api sample" style="width: 65%; display: block">
</details>

### [finetune](https://www.tigerbot.com/api-reference/finetune)

<details><summary><b>Example</b></summary>
<img src="image/api/demo/finetune.png" alt="tigerbot chat-api sample" style="width: 65%; display: block">
</details>

### [embedding](https://www.tigerbot.com/api-reference/embedding)

<details><summary><b>Example</b></summary>
<img src="image/api/demo/embedding.png" alt="tigerbot chat-api sample" style="width: 65%; display: block">
</details>

### [summarization](https://www.tigerbot.com/api-reference/summarization)

<details><summary><b>Example</b></summary>
<img src="image/api/demo/summarization.png" alt="tigerbot chat-api sample" style="width: 65%; display: block">
</details>

### [pdf2text](https://www.tigerbot.com/api-reference/pdf2text)

<details><summary><b>Example</b></summary>
<img src="image/api/demo/pdf2text.png" alt="tigerbot chat-api sample" style="width: 65%; display: block">
</details>

### [text2image](https://www.tigerbot.com/api-reference/text2image)

<details><summary><b>Example</b></summary>
<img src="image/api/demo/text2image.png" alt="tigerbot chat-api sample" style="width: 65%; display: block">
</details>
</details>

## Others

<details><summary><b>User cases</b></summary>

![image](./image/api/case-1.png)
![image](image/api/case-2.png)
![image](image/api/case-3.png)
![image](image/api/case-4.png)
![image](image/api/case-5.png)
![image](image/api/case-6.png)

</details>

<details><summary><b>Community</b></summary>

- [Build desktop chatbot application with Tigerbot and Svelte fast](https://github.com/SaraiQX/tigerbot-svelte-app)
- [QQ bot based on  Tigerbot with custom knowledge base](https://github.com/wordweb/Tiger-qq-bot)
- [Application based on  Tigerbot with custom knowledge base](https://github.com/wordweb/langchain-ChatGLM-and-TigerBot)
- [Run TigerBot on Colab](https://github.com/runfuture/tigerbot/blob/main/test_tigerbot_7b_sft_4bit_128g.ipynb)
- [Run TigerBot on Windows](https://www.bilibili.com/video/BV1Ru411a7Kq/)
</details>

<details><summary><b>Join us</b></summary>

#### Our product

https://www.tigerbot.com

#### Tel us

021-63888086

#### Email us

<p>cong.fu@tigerbot.com</p>
<p>wei.cai@tigerbot.com</p>

#### Wechat

<img src="image/qiyewechat.png" alt="Tiger" style="width: 260px;  "></a>

## Limitations and Disclaimers
Current models may contain hallucinatory, misleading, or discriminatory content. 
Please use the content generated by TigerBot series models with caution, and do not spread the generated harmful content.

The project developer is not responsible for any harm or loss caused by the use of this project 
(including but not limited to data, models, codes, etc.).
