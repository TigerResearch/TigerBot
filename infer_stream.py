import os

import fire
import torch
import readline
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from transformers import AutoTokenizer
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.generation.streamers import TextIteratorStreamer
from threading import Thread
from typing import List, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

max_generate_length: int = 1024


def get_prompt(query, history=None):
    if not history:
        prompt = "\n\n### Instruction:\n{}\n\n### Response:\n".format(query)
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "\n\n### Instruction:\n{}\n\n### Response:\n{}".format(old_query, response)
        prompt += "\n\n### Instruction:\n{}\n\n### Response:\n".format(query)
    return prompt


def get_model(model):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = BloomForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    return model


def stream_chat(
        target,
        tokenizer,
        input,
        history: List[Tuple[str, str]] = None,
        top_p: float = 0.95,
        temperature: float = 0.8,
        max_input_length: int = 512,
        max_generate_length: int = 1024,
):
    generation_kwargs = {
        "top_p": top_p,
        "temperature": temperature,
        "max_length": max_generate_length,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "early_stopping": True,
        "no_repeat_ngram_size": 4,
    }
    global model
    query_text = input.strip()
    input_text = get_prompt(query_text, history)
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=max_input_length)
    device = torch.cuda.current_device()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    streamer = TextIteratorStreamer(tokenizer=tokenizer)
    kwargs = dict(inputs, streamer=streamer, **generation_kwargs)
    thread = Thread(target=target, kwargs=kwargs)
    thread.start()
    generated_text = ""
    new_response = ""
    for new_text in streamer:
        if len(new_text) == 0:
            continue
        generated_text += new_text
        if len(generated_text) > len(input_text):
            response = new_text.rstrip("</s>")
            new_response += response
            new_history = history + [(query_text, new_response)]
            yield new_response, new_history


def main(
    model_path: str,
    max_input_length: int = 512,
    max_generate_length: int = 1024,
):
    print(f"loading model: {model_path}...")
    model = get_model(model_path)
    max_memory = get_balanced_memory(model)
    device_map = infer_auto_device_map(model, max_memory=max_memory,
                                       no_split_module_classes=["BloomBlock"])
    print("Using the following device map for the model:", device_map)
    model = dispatch_model(model, device_map=device_map, offload_buffers=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        model_max_length=max_generate_length,
        padding_side="left",
        truncation_side='left',
        padding=True,
        truncation=True
    )
    if tokenizer.model_max_length is None or tokenizer.model_max_length > 1024:
        tokenizer.model_max_length = 1024
    history = []
    while True:
        raw_text = input("prompt(\"exit\" to end, \"clear\" to clear session) >>> ")
        if not raw_text:
            print('prompt should not be empty!')
            continue
        if raw_text.strip() == "exit":
            print('session ended.')
            break
        if raw_text.strip() == "clear":
            print('session cleared.')
            history = []
            continue
        print("=" * 100)
        res_len = 0
        for (res, history) in stream_chat(
            model.generate,
            tokenizer,
            raw_text,
            history,
            max_input_length=max_input_length,
            max_generate_length=max_generate_length
        ):
            print(res[res_len:], end="")
            res_len = len(res)
        print("")
        print("=" * 100)


if __name__ == "__main__":
    fire.Fire(main)
