import argparse
import os
from dataclasses import dataclass, field
from os.path import join
from typing import *

import bitsandbytes as bnb
import datasets
import evaluate
import torch
import transformers
from peft import get_peft_model, PeftModel, prepare_model_for_kbit_training, LoraConfig
from peft.tuners.lora import LoraLayer
from transformers import BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

os.environ["WANDB_DISABLED"] = "true"


@dataclass
class PeftConfig:
    model_name_or_path: Optional[str] = field(metadata={"help": "Path to pretrained model checkpoint"})
    data_files: Optional[str] = field(default=None, metadata={"help": "Local data files"})
    max_length: int = field(default=1024, metadata={
        "help": "Maximum source + target sequence length. Sequences will be right padded (and possibly truncated)."}, )
    preprocess_num_workers: int = field(default=4,
                                        metadata={"help": "The number of processes to use for the preprocessing."})
    pad_to_max_length: bool = field(default=True, metadata={
        "help": "Pad all examples to max_length. This is for fair comparison between different batch size"})

    # Lora related param here
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(default=True,
                               metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4",
                            metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    full_finetune: bool = field(default=False, metadata={"help": "Finetune the entire model without adapters."})
    do_infer: bool = field(default=False, metadata={"help": "Finetune the entire model without adapters."})


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model, bits):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if bits == 4:
        trainable_params /= 2

    if all_param != 0:
        print(
            f"trainable params: {trainable_params} || "
            f"all params: {all_param} || "
            f"trainable: {100 * trainable_params / (all_param):.2f}%"
        )


def get_accelerate_model(args, checkpoint_dir):
    device_map = "auto"
    if args.full_finetune:
        assert args.bits in [16, 32]

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    print(f"compute_type: {compute_dtype}")

    if args.full_finetune:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type,
            ),
            torch_dtype=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type,
            ),
            torch_dtype=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        )
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('=' * 80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('=' * 80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not args.full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'),
                                              is_trainable=not args.do_infer)
            if args.do_infer:
                print("Merge adapter weights to base model.")
                model = model.merge_and_unload()
        else:
            print(f'Adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=preds, references=labels)


def train():
    parser = transformers.HfArgumentParser((PeftConfig, transformers.TrainingArguments))
    peft_args, train_args = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(peft_args), **vars(train_args))

    # Load model
    model = get_accelerate_model(args, None)
    print_trainable_parameters(model, args.bits)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='right',
                                                           trunction_side="right", max_length=args.max_length)

    # Load dataset
    train_ds, validation_ds = datasets.load_dataset('json', data_files=args.data_files,
                                                    split=['train[:80%]', 'train[80%:]'])
    raw_datasets = datasets.DatasetDict({"train": train_ds, "validation": validation_ds})

    def process_supervised(record):
        input_s = record['instruction'] + '\n' + (record.get('input', '') or '')
        output_s = record['output']
        tokenized = tokenizer([input_s, output_s])
        token_ids = [tok_id for tok_ids in tokenized['input_ids'] for tok_id in tok_ids]
        attention_mask = [mask for masks in tokenized['attention_mask'] for mask in masks]
        if token_ids[-1] != tokenizer.eos_token_id:
            token_ids += [tokenizer.eos_token_id]
            attention_mask += [1]
        processed_record = {
            "input_ids": token_ids[:args.max_length],
            "attention_mask": attention_mask[:args.max_length],
            "labels": token_ids.copy()[:args.max_length]
        }
        if args.pad_to_max_length:
            processed_record = {
                "input_ids": processed_record["input_ids"] + [tokenizer.pad_token_id] * (
                        args.max_length - len(processed_record["input_ids"])),
                "attention_mask": processed_record["attention_mask"] + [0] * (
                        args.max_length - len(processed_record["attention_mask"])),
                "labels": processed_record["labels"] + [-100] * (args.max_length - len(processed_record["labels"]))
            }

        # ignore input label, label is ignored if value is -100
        processed_record["labels"][:min(len(tokenized["input_ids"][0]), args.max_length)] = [-100] * min(
            len(tokenized["input_ids"][0]), args.max_length)
        return {k: torch.tensor(v, dtype=torch.int) for k, v in processed_record.items()}

    with train_args.main_process_first(desc="Process supervised dataset"):
        sft_dataset = raw_datasets.map(
            process_supervised,
            batched=False,
            num_proc=args.preprocess_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            desc="Process supervised dataset"
        )

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=sft_dataset["train"],
        eval_dataset=sft_dataset["validation"],
        data_collator=transformers.DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest",
                                                                      max_length=args.max_length,
                                                                      label_pad_token_id=-100)
    )

    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    all_metrics = {"run_name": args.run_name}
    if args.do_train:
        print("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    if args.do_eval:
        print("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)


if __name__ == "__main__":
    train()
