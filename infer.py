import os
from typing import Optional

import fire
import torch
import transformers
from packaging import version

from utils.modeling_hack import get_model
from utils.streaming import generate_stream
from builtins import True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tok_ins = "\n\n### Instruction:\n"
tok_res = "\n\n### Response:\n"
prompt_input = tok_ins + "{instruction}" + tok_res
bos_token = "<|begin_of_text|>"
eos_token = "<|end_of_text|>"


def main(
        model_path: str,
        max_input_length: int=512,
        max_generate_length: int=1024,
        model_type: str='chat',
        rope_scaling: Optional[str]=None,
        rope_factor: float=8.0,
        streaming: bool=True  # streaming is always enabled now
):
    assert version.parse(transformers.__version__) >= version.parse("4.34")
    assert model_type.lower() in ['chat', 'base'], f"model_type must be one of ['chat', 'base'], got {model_type}"
    assert rope_scaling in [None, 'yarn',
                            'dynamic'], f"rope_scaling must be one of [None, 'yarn', 'dynamic'], got {rope_scaling}"

    model, tokenizer, generation_config = get_model(model_path=model_path, rope_scaling=rope_scaling,
                                                    rope_factor=rope_factor)
    generation_config.max_new_tokens = max_generate_length
    generation_config.max_length = max_input_length + max_generate_length
    
    # sampling gen configs
    generation_config.do_sample = True
    generation_config.temperature = 0.6
    # generation_config.top_k = 5
    # generation_config.top_p = 0.9
    generation_config.repetition_penalty = 1.08
    generation_config.use_cache = True

    device = torch.cuda.current_device()
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
        if model_type == 'chat':
            tokenizer.add_bos_token = False
            tokenizer.add_eos_token = False
            input_text = prompt_input.format_map({'instruction': sess_text.split(tok_ins, 1)[1]})
        else:
            tokenizer.add_bos_token = True
            tokenizer.add_eos_token = False
            input_text = query_text
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=max_input_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print('=' * 100)
        answer = ''
        for text in generate_stream(model, tokenizer, inputs['input_ids'], inputs['attention_mask'],
                                    generation_config=generation_config):
            print(text, end='', flush=True)
            answer += text
        sess_text += tok_res + answer
        print('')
        print("=" * 100)


if __name__ == "__main__":
    fire.Fire(main)
