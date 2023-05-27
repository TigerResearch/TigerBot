import argparse

import torch
from gptq import quant

from utils import find_layers, DEV
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, modeling_utils
from accelerate.utils import get_balanced_memory
from accelerate import infer_auto_device_map, dispatch_model


def get_model(model):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = 2048
    return model


def load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    config = AutoConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)

    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    model.seqlen = 2048
    print('Done.')

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='llama model to load'
    )
    parser.add_argument(
        '--wbits', type=int, default=4, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--load', type=str,
        help='Load quantized model.'
    )

    parser.add_argument(
        '--max_input_length', type=int, default=1024,
        help='The maximum length of the input prompt.'
    )

    parser.add_argument(
        '--max_generate_length', type=int, default=2048,
        help='The maximum length the generated tokens can have. Corresponds to the length of the input prompt + max_new_tokens'
    )

    parser.add_argument(
        '--top_p', type=float, default=0.95,
        help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.'
    )

    parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='The value used to module the next token probabilities.'
    )
    parser.add_argument(
        '--no_repeat_ngram_size', type=int, default=4,
        help=' If set to int > 0, all ngrams of that size can only occur once.'
    )

    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        model = get_model(args.model)

    max_memory = get_balanced_memory(model)
    device_map = infer_auto_device_map(model, max_memory=max_memory,
                                       no_split_module_classes=["BloomBlock"])
    print("Using the following device map for the model:", device_map)
    model = dispatch_model(model, device_map=device_map, offload_buffers=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", truncation_side='left')
    tok_ins = "\n\n### Instruction:\n"
    tok_res = "\n\n### Response:\n"
    prompt_input = tok_ins + "{instruction}" + tok_res

    max_input_length = args.max_input_length
    max_generate_length = args.max_generate_length
    generation_kwargs = {
        "top_p": args.top_p,
        "temperature": args.temperature,
        "max_length": max_generate_length,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "early_stopping": True,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
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
        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=max_input_length).to(
            DEV)
        input_length = input_ids.shape[1]
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                **generation_kwargs
            )
        result = tokenizer.decode([el.item() for el in generated_ids[0][input_length:]], skip_special_tokens=True,
                                  spaces_between_special_tokens=False)
        answer = result.rstrip(tokenizer.eos_token)
        sess_text += tok_res + answer
        print("=" * 100)
        print(answer)
        print("=" * 100)
