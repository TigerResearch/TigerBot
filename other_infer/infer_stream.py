import os
import sys

import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

max_generate_length: int = 1024


def main(
        model_path: str,
        max_input_length: int = 512,
        max_generate_length: int = 1024,
):
    print(f"loading model: {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')

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
        for (res, history_) in model.stream_chat(
                tokenizer,
                raw_text,
                history,
                max_input_length=max_input_length,
                max_generate_length=max_generate_length
        ):
            if res is not None:
                print("\r" + res, end="")
            if history_ is not None:
                history = history_
        print("")
        print("=" * 100)


if __name__ == "__main__":
    fire.Fire(main)
