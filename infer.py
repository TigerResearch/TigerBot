from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate.utils import get_balanced_memory
from accelerate import infer_auto_device_map, dispatch_model
import torch
import fire

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tok_ins = "\n\n### Instruction:\n"
tok_res = "\n\n### Response:\n"
prompt_input = tok_ins + "{instruction}" + tok_res


def get_model(model):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    return model


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

    device = torch.cuda.current_device()

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

    generation_kwargs = {
        "top_p": 0.95,
        "temperature": 0.8,
        "max_length": max_generate_length,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "early_stopping": True,
        "no_repeat_ngram_size": 4,
    }

    sess_text = ""
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
            sess_text = ""
            continue

        query_text = raw_text.strip()
        sess_text += tok_ins + query_text
        input_text = prompt_input.format_map({'instruction': sess_text.split(tok_ins, 1)[1]})
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=max_input_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model.generate(**inputs, **generation_kwargs)
        result = tokenizer.decode(output[0], skip_special_tokens=False, spaces_between_special_tokens=False)
        answer = result.rsplit(tok_res, 1)[1].rstrip(tokenizer.eos_token)
        sess_text += tok_res + answer

        print("=" * 100)
        print(answer)
        print("=" * 100)


if __name__ == "__main__":
    fire.Fire(main)