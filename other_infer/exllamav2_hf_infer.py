import os
import time
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Optional, Union

import fire
import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config
from torch.nn import CrossEntropyLoss
from transformers import (
    GenerationConfig,
    LlamaTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    TextIteratorStreamer,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tok_ins = "\n\n### Instruction:\n"
tok_res = "\n\n### Response:\n"
prompt_input = tok_ins + "{instruction}" + tok_res


def progress_rep(module, num_modules):
    yield 100 * module / num_modules


class Exllamav2HF(PreTrainedModel):
    def __init__(self, config: ExLlamaV2Config):
        super().__init__(PretrainedConfig())
        self.ex_config = config
        self.ex_model = ExLlamaV2(config)

        self.generation_config = GenerationConfig()

        self.ex_cache = ExLlamaV2Cache(self.ex_model, lazy=True)
        f = self.ex_model.load_autosplit_gen(
            self.ex_cache, last_id_only=True, callback_gen=progress_rep
        )
        for _ in f:
            pass
        self.past_seq = None

    def _validate_model_class(self):
        pass

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        pass

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    @property
    def device(self) -> torch.device:
        return torch.device(0)

    def __call__(self, *args, **kwargs):
        use_cache = kwargs.get("use_cache", True)
        labels = kwargs.get("labels", None)
        past_key_values = kwargs.get("past_key_values", None)

        if len(args) > 0:
            input_ids = args[0]
            is_negative = True
            past_seq = self.past_seq_negative
            ex_cache = None
        else:
            input_ids = kwargs["input_ids"]
            is_negative = False
            past_seq = self.past_seq
            ex_cache = self.ex_cache

        seq = input_ids[0].tolist()
        if is_negative and past_key_values is not None:
            seq = past_key_values + seq

        seq_tensor = torch.tensor(seq)
        reset = True

        # Make the forward call
        if labels is None:
            if past_seq is not None:
                min_length = min(past_seq.shape[0], seq_tensor.shape[0])
                indices = torch.nonzero(
                    ~torch.eq(
                        past_seq[:min_length], seq_tensor[:min_length]
                    )
                )
                if len(indices) > 0:
                    longest_prefix = indices[0].item()
                else:
                    longest_prefix = min_length

                if longest_prefix > 0:
                    reset = False
                    ex_cache.current_seq_len = longest_prefix
                    if len(seq_tensor) - longest_prefix > 1:
                        self.ex_model.forward(
                            seq_tensor[longest_prefix:-1].view(1, -1),
                            ex_cache,
                            preprocess_only=True,
                        )

            if reset:
                ex_cache.current_seq_len = 0
                if len(seq_tensor) > 1:
                    self.ex_model.forward(
                        seq_tensor[:-1].view(1, -1),
                        ex_cache,
                        preprocess_only=True,
                    )

            logits = self.ex_model.forward(
                seq_tensor[-1:].view(1, -1), ex_cache
            ).to(input_ids.device)
        else:
            ex_cache.current_seq_len = 0
            logits = self.ex_model.forward(
                seq_tensor.view(1, -1), ex_cache, last_id_only=False
            )

        if is_negative:
            self.past_seq_negative = seq_tensor
        else:
            self.past_seq = seq_tensor

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=seq if use_cache else None,
            loss=loss,
        )

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[
                Union[str, os.PathLike]
            ],
            *model_args,
            **kwargs,
    ):
        assert (
                len(model_args) == 0 and len(kwargs) == 0
        ), "extra args is currently not supported"
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(
                pretrained_model_name_or_path
            )

        config = ExLlamaV2Config()
        config.model_dir = str(pretrained_model_name_or_path)
        config.prepare()

        # config.max_seq_len = shared.args.max_seq_len
        # config.scale_pos_emb = shared.args.compress_pos_emb
        # config.scale_alpha_value = shared.args.alpha_value

        return Exllamav2HF(config)


def get_model(model_path):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    print(f"Loading model from {model_path}...")
    model = Exllamav2HF.from_pretrained(model_path)
    model.eval()
    print("Done")

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    print("Done")

    print(f"Loading generation config from {model_path}...")
    generation_config = GenerationConfig.from_pretrained(model_path)
    print("Done")
    return model, tokenizer, generation_config


def main(
        model_path: str,
        max_input_length: int = 512,
        max_generate_length: int = 2048,
        stream: bool = True,
):
    print(f"loading model: {model_path}...")

    model = get_model(model_path)
    device = torch.cuda.current_device()
    model, tokenizer, generation_config = get_model(model_path)

    generation_config.do_sample = False
    generation_config.max_length = max_input_length + max_generate_length
    generation_config.max_new_tokens = max_generate_length

    sess_text = ""

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
    )
    generation_kwargs = generation_config.to_dict()

    def eval_generate(**args):
        with torch.inference_mode(mode=True):
            model.eval()
            model.generate(**args)

    with torch.inference_mode(mode=True):
        model.eval()
        while True:
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
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length,
            )
            tic = time.perf_counter()
            if stream:
                generation_kwargs["streamer"] = streamer
                for k, v in inputs.items():
                    generation_kwargs[k] = v.to(device)
                thread = Thread(
                    target=eval_generate, kwargs=generation_kwargs
                )
                thread.start()
                answer = ""
                flag = False
                print("=" * 100)
                for new_text in streamer:
                    if new_text.endswith(tokenizer.eos_token):
                        new_text = new_text.rsplit(
                            tokenizer.eos_token, 1
                        )[0].strip()
                        flag = True
                    print(new_text, end="")
                    answer += new_text
                    if flag:
                        break
                toc = time.perf_counter()
                num_tok = len(tokenizer.encode(answer))
            else:
                if "streamer" in generation_kwargs:
                    del generation_kwargs["streamer"]
                inputs = {k: v.to(device) for k, v in inputs.items()}
                output = model.generate(
                    **inputs, **generation_config.to_dict()
                )
                toc = time.perf_counter()
                num_tok = output.shape[1]
                output_str = tokenizer.decode(
                    output[0],
                    skip_special_tokens=False,
                    spaces_between_special_tokens=False,
                )
                answer = output_str.rsplit(tok_res, 1)[1].strip()
                if answer.endswith(tokenizer.eos_token):
                    answer = answer.rsplit(tokenizer.eos_token, 1)[
                        0
                    ].strip()
                print(answer)

            sess_text += tok_res + answer
            res_time = toc - tic
            print(
                f"\n[time: {res_time:0.4f} sec, speed: {num_tok / res_time:0.4f} tok/sec]"
            )
            print("=" * 100)


if __name__ == "__main__":
    fire.Fire(main)
