from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import fire

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(
    model_path: str="TigerResearch/tigerbot-7b-base",
    max_input_length: int=512,
    max_generate_length: int=1024,
):
    print(f"loading model: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        model_max_length=max_generate_length,
        padding_side="left",
        truncation_side='left',
        padding=True,
        truncation=True,
        # use_fast=False,
    )
    if tokenizer.model_max_length is None or tokenizer.model_max_length > 1024:
        tokenizer.model_max_length = 1024
    
    model = AutoModelForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.pad_token_id, torch_dtype=torch.bfloat16)
    device = torch.cuda.current_device()
    model.to(device)
    
    generation_kwargs = {
        "top_p": 0.95,
        "temperature": 0.8,
        "max_length": max_generate_length,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "early_stopping": True,
        "no_repeat_ngram_size": 4,
        }

    while True:
        raw_text = input("prompt(\"exit\" to end, \"clear\" to clear session) >>> ")
        if not raw_text:
            print('prompt should not be empty!')
            continue
        if raw_text.strip() == "exit":
            print('session ended.')
            break
    
        query_text = raw_text.strip()
        inputs = tokenizer(query_text, return_tensors='pt', truncation=True, max_length=max_input_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model.generate(**inputs, **generation_kwargs)
        result = tokenizer.decode(output[0], skip_special_tokens=False, spaces_between_special_tokens=False)
        answer = result.rstrip(tokenizer.eos_token)    
        print("=" * 100)
        print(answer)
        print("=" * 100)


if __name__ == "__main__":
    fire.Fire(main)

