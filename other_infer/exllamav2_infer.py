import os
import sys
import time

import fire
import torch
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2Sampler,
    ExLlamaV2StreamingGenerator,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tok_ins = "\n\n### Instruction:\n"
tok_res = "\n\n### Response:\n"
prompt_input = tok_ins + "{instruction}" + tok_res


def main(
    model_path: str,
    max_length: int = 1024,
    rope_scale: float = 1.0,
    rope_alpha: float = 1.0,
    no_flash_attn: bool = True,
    temperature: float = 0.3,
    repetition_penalty: float = 1.1,
    gpu_split=None,
):
    # Create config
    config = ExLlamaV2Config()
    config.model_dir = model_path
    config.prepare()

    # Set config options
    config.max_seq_len = max_length
    config.rope_scale = rope_scale
    config.rope_alpha = rope_alpha
    config.no_flash_attn = no_flash_attn

    # Load model
    model = ExLlamaV2(config)

    split = None
    if gpu_split:
        split = [float(alloc) for alloc in gpu_split.split(",")]
    model.load(split)

    # Load tokenizer
    tokenizer = ExLlamaV2Tokenizer(config)

    # Create cache
    cache = ExLlamaV2Cache(model)

    # Generator
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = temperature
    settings.token_repetition_penalty = repetition_penalty

    # Stop conditions
    generator.set_stop_conditions([tokenizer.eos_token_id])

    # Main loop
    sess_text = ""

    while True:
        # Get user prompt
        raw_text = input(
            'prompt("exit" to end, "clear" to clear session) >>> '
        )
        if not raw_text:
            print("prompt should not be empty!")
            continue
        if raw_text.strip() == "exit":
            print("session ended.")
            break
        if raw_text.strip() == "clear":
            print("session cleared.")
            sess_text = ""
            continue

        query_text = raw_text.strip()
        sess_text += tok_ins + query_text
        input_text = prompt_input.format_map(
            {"instruction": sess_text.split(tok_ins, 1)[1]}
        )

        active_context = tokenizer.encode(input_text, add_bos=True)
        generator.begin_stream(active_context, settings)

        response_tokens = 0
        response_text = ""

        print("=" * 100)
        tic = time.perf_counter()
        while True:
            # Get response stream
            chunk, eos, tokens = generator.stream()
            if len(response_text) == 0:
                chunk = chunk.lstrip()
            response_text += chunk
            print(chunk, end="")
            sys.stdout.flush()
            response_tokens += 1

            # If model has run out of space, rebuild the context and restart stream
            # if generator.full():
            #     generator.begin_stream(active_context, settings)

            # EOS signal returned
            if tok_ins in response_text:
                response_text = response_text.split(tok_ins)[0]
                print(response_text, end="")
                sys.stdout.flush()
                break

            if tok_res in response_text:
                response_text = response_text.split(tok_res)[0]
                print(response_text, end="")
                sys.stdout.flush()
                break

            if eos:
                break

        toc = time.perf_counter()
        res_time = toc - tic
        print(
            f"\n[time: {res_time:0.4f} sec, speed: {response_tokens / res_time:0.4f} tok/sec]"
        )
        print("=" * 100)
        sess_text += tok_res + response_text


if __name__ == "__main__":
    fire.Fire(main)
