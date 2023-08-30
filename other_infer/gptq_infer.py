import time
import os
import random
import logging
import fire

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tok_ins = "\n\n### Instruction:\n"
tok_res = "\n\n### Response:\n"
prompt_input = tok_ins + "{instruction}" + tok_res


def get_model(model_name_or_path, model_basename):
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    quantize_config = BaseQuantizeConfig.from_pretrained(model_name_or_path)
    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                                               model_basename=model_basename,
                                               use_safetensors=True,
                                               device_map='auto',
                                               use_triton=False,
                                               quantize_config=quantize_config)
    return model


def main(model_path: str = "TigerResearch/tigerbot-13b-chat-8bit",
         model_basename: str = "tigerbot_13b_chat_8bit_128g",
         max_input_length: int = 512,
         max_generate_length: int = 1024):
    model = get_model(model_name_or_path=model_path, model_basename=model_basename)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=max_generate_length,
        padding_side="left",
        truncation_side='left',
        padding=True,
        truncation=True
    )
    if tokenizer.model_max_length is None or tokenizer.model_max_length > max_generate_length:
        tokenizer.model_max_length = max_generate_length
    generation_config = GenerationConfig.from_pretrained(model_path)

    generation_config.max_length = max_generate_length
    print(generation_config)

    device = torch.cuda.current_device()

    sess_text = ""
    while True:
        raw_text = input(
            "prompt(\"exit\" to end, \"clear\" to clear session) >>> ")
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
        input_text = prompt_input.format_map(
            {'instruction': sess_text.split(tok_ins, 1)[1]})
        inputs = tokenizer(
            input_text, return_tensors='pt', truncation=True, max_length=max_input_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model.generate(**inputs, **generation_config.to_dict())
        output_str = tokenizer.decode(
            output[0], skip_special_tokens=False, spaces_between_special_tokens=False)
        answer = output_str.rsplit(tok_res, 1)[1].strip()
        if answer.endswith(tokenizer.eos_token):
            answer = answer.rsplit(tokenizer.eos_token, 1)[0].strip()
        sess_text += tok_res + answer

        print("=" * 100)
        print(answer)
        print("=" * 100)


if __name__ == "__main__":
    fire.Fire(main)