import os
import sys
import time

import fire
from transformers import GenerationConfig
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tok_ins = "\n\n### Instruction:\n"
tok_res = "\n\n### Response:\n"
prompt_input = tok_ins + "{instruction}" + tok_res


def main(
    model_path: str,
    temperature: float = 0.3,
    repetition_penalty: float = 1.1,
    max_generate_length: int = 2048,
):
    # Create config
    config = ExLlamaV2Config()
    config.model_dir = model_path
    config.prepare()

    max_new_tokens = max_generate_length

    # Load model
    model = ExLlamaV2(config)
    print("Loading model: " + model_path)

    cache = ExLlamaV2Cache(model, lazy = True)
    model.load_autosplit(cache)

    # Load tokenizer
    tokenizer = ExLlamaV2Tokenizer(config)
    # tokenizer.

    # Initialize generator
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)


    # Settings
    generation_config = GenerationConfig.from_pretrained(model_path)
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = generation_config.temperature
    settings.token_repetition_penalty = generation_config.repetition_penalty

    # Generator
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

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

        input_ids = tokenizer.encode(input_text, encode_special_tokens = True)

        # Make sure CUDA is initialized so we can measure performance
        generator.warmup()

        print("=" * 100)
        tic = time.perf_counter()

        sys.stdout.flush()

        generator.set_stop_conditions([60513, 60512, tokenizer.eos_token_id, tokenizer.pad_token_id])
        generator.begin_stream(input_ids, settings)

        # Streaming loop. Note that repeated calls to sys.stdout.flush() adds some latency, but some
        # consoles won't update partial lines without it.
        generated_tokens = 0

        answer = ''
        while True:
            chunk, eos, _ = generator.stream()
            generated_tokens += 1
            answer += chunk
            print (chunk, end = "")
            sys.stdout.flush()
            if eos or generated_tokens == max_new_tokens: break

        toc = time.perf_counter()
        res_time = toc - tic
        print(
            f"\n[time: {res_time:0.4f} sec, speed: {generated_tokens / res_time:0.4f} tok/sec]"
        )
        print("=" * 100)
        sess_text += tok_res + answer


if __name__ == "__main__":
    fire.Fire(main)

