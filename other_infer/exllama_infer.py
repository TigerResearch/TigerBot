import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import fire
import torch
from exllama_lib.model import ExLlama, ExLlamaCache, ExLlamaConfig
from torch.nn import CrossEntropyLoss
from transformers import (
    GenerationConfig,
    LlamaTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tok_ins = "\n\n### Instruction:\n"
tok_res = "\n\n### Response:\n"
prompt_input = tok_ins + "{instruction}" + tok_res


class ExllamaHF(PreTrainedModel):
    def __init__(self, config: ExLlamaConfig):
        super().__init__(PretrainedConfig())
        self.ex_config = config
        self.ex_model = ExLlama(self.ex_config)
        self.generation_config = GenerationConfig()
        self.lora = None

        self.ex_cache = ExLlamaCache(self.ex_model)
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
            ex_cache = self.ex_cache_negative
        else:
            input_ids = kwargs["input_ids"]
            is_negative = False
            past_seq = self.past_seq
            ex_cache = self.ex_cache

        seq = input_ids[0].tolist()
        if is_negative and past_key_values is not None:
            seq = past_key_values + seq

        seq_tensor = torch.tensor(seq)

        # Make the forward call
        if labels is None:
            if past_seq is None or not torch.equal(
                past_seq, seq_tensor[:-1]
            ):
                ex_cache.current_seq_len = 0
                self.ex_model.forward(
                    torch.tensor([seq[:-1]], dtype=torch.long),
                    ex_cache,
                    preprocess_only=True,
                    lora=self.lora,
                )

            logits = self.ex_model.forward(
                torch.tensor([seq[-1:]], dtype=torch.long),
                ex_cache,
                lora=self.lora,
            ).to(input_ids.device)
        else:
            ex_cache.current_seq_len = 0
            logits = self.ex_model.forward(
                torch.tensor([seq], dtype=torch.long),
                ex_cache,
                last_id_only=False,
                lora=self.lora,
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

        config = ExLlamaConfig(
            pretrained_model_name_or_path / "config.json"
        )

        weight_path = None
        for ext in [".safetensors", ".pt", ".bin"]:
            found = list(pretrained_model_name_or_path.glob(f"*{ext}"))
            if len(found) > 0:
                weight_path = found[-1]
                break
        assert (
            weight_path is not None
        ), f'could not find weight in "{pretrained_model_name_or_path}"'

        config.model_path = str(weight_path)
        config.max_seq_len = 2048
        config.compress_pos_emb = 1

        if torch.version.hip:
            config.rmsnorm_no_half2 = True
            config.rope_no_half2 = True
            config.matmul_no_half2 = True
            config.silu_no_half2 = True

        # This slowes down a bit but align better with autogptq generation.
        # TODO: Should give user choice to tune the exllama config
        # config.fused_attn = False
        # config.fused_mlp_thd = 0

        return ExllamaHF(config)


def get_model(model):
    # from accelerate import infer_auto_device_map, dispatch_model
    # from accelerate.utils import get_balanced_memory

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # model = AutoGPTQForCausalLM.from_quantized(self.model_path,
    #                                            model_basename=self.model_basename,
    #                                            use_safetensors=True,
    #                                            trust_remote_code=True,
    #                                            device_map='auto',
    #                                            use_triton=self.use_triton,
    #                                            quantize_config=quantize_config)

    model = ExllamaHF.from_pretrained(model)

    # DONOT SUPPORT ACCELERATE !!!
    # max_memory = get_balanced_memory(model)
    # device_map = infer_auto_device_map(model, max_memory=max_memory,
    #                                 no_split_module_classes=[])
    # print("Using the following device map for the model:", device_map)
    # model = dispatch_model(model, device_map=device_map, offload_buffers=True)
    return model


def main(
    model_path: str,
    max_input_length: int = 512,
    max_generate_length: int = 1024,
):
    print(f"loading model: {model_path}...")

    model = get_model(model_path)
    # max_memory = get_balanced_memory(model)
    # device_map = infer_auto_device_map(model, max_memory=max_memory,
    #                                    no_split_module_classes=[])
    # print("Using the following device map for the model:", device_map)
    # model = dispatch_model(model, device_map=device_map, offload_buffers=True)

    device = torch.cuda.current_device()

    tokenizer = LlamaTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        model_max_length=max_generate_length,
        padding_side="left",
        truncation_side="left",
        padding=True,
        truncation=True,
    )
    if (
        tokenizer.model_max_length is None
        or tokenizer.model_max_length > max_generate_length
    ):
        tokenizer.model_max_length = max_generate_length

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
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model.generate(**inputs, **generation_kwargs)
        output_str = tokenizer.decode(
            output[0],
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )
        answer = output_str.rsplit(tok_res, 1)[1].strip()
        if answer.endswith(tokenizer.eos_token):
            answer = answer.rsplit(tokenizer.eos_token, 1)[0].strip()
        sess_text += tok_res + answer

        print("=" * 100)
        print(answer)
        print("=" * 100)


if __name__ == "__main__":
    fire.Fire(main)
