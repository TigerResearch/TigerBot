import os
import readline

import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

print(readline)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tok_ins = "\n\n### Instruction:\n"
tok_res = "\n\n### Response:\n"
prompt_input = tok_ins + "{instruction}" + tok_res


def main(
        model_path: str,
        max_input_length: int = 512,
        max_generate_length: int = 1024,
        model_type: str = 'chat',
        use_flash_attn: bool = False
):
    if model_type.lower() not in ['chat', 'base']:
        raise ValueError(f"model_type must be one of ['chat', 'base'], got {model_type}")
    if use_flash_attn:
        from flash_attention import replace_attn_with_flash_attn
        replace_attn_with_flash_attn()
        print("using flash attention...")

    print(f"loading model: {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')

    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.max_length = max_generate_length
    print(generation_config)

    device = torch.cuda.current_device()

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
            input_text = prompt_input.format_map({'instruction': sess_text.split(tok_ins, 1)[1]})
        else:
            input_text = query_text
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=max_input_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model.generate(**inputs, **generation_config.to_dict())
        answer = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False,
                                  spaces_between_special_tokens=False)
        if answer.endswith(tokenizer.eos_token):
            answer = answer.rsplit(tokenizer.eos_token, 1)[0].strip()

        sess_text += tok_res + answer

        print("=" * 100)
        print(answer)
        print("=" * 100)


if __name__ == "__main__":
    fire.Fire(main)
