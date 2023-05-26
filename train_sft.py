import os
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import torch
import transformers

os.environ["WANDB_DISABLED"] = "true"


@dataclass
class SFTConfig:
    model_path: Optional[str] = field(metadata={"help": "Path to pretrained model checkpoint"})
    train_file_path: Optional[str] = field(metadata={"help": "Path to train data file/directory"})
    validate_file_path: Optional[str] = field(default=None, metadata={"help": "Path to validation data file/directory"})
    max_length: int = field(default=1024, metadata={"help": "Max length of input"})
    text_key_name: Optional[str] = field(default="content",
                                         metadata={"help": "key to text field name in train and validation file"})
    preprocess_num_workers: int = field(default=8,
                                        metadata={"help": "The number of processes to use for the preprocessing."})


def check_file_exist(path: str):
    if not os.path.exists(path):
        raise ValueError(f"Path: {path} not exists!")


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


def main():
    transformers.set_seed(1234)
    parser = transformers.HfArgumentParser((SFTConfig, transformers.TrainingArguments))
    sft_config, training_args = parser.parse_args_into_dataclasses()

    # check file existence
    check_file_exist(sft_config.model_path)
    if sft_config.train_file_path:
        check_file_exist(sft_config.train_file_path)
    if sft_config.validate_file_path:
        check_file_exist(sft_config.validate_file_path)

    # load model, tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(sft_config.model_path, padding_side='right',
                                                           trunction_side="right",
                                                           max_length=sft_config.max_length)

    model = transformers.AutoModelForCausalLM.from_pretrained(sft_config.model_path)

    # Split 20% of train data as validation data
    if not sft_config.validate_file_path:
        train_ds, validation_ds = datasets.load_dataset('json', data_files=sft_config.train_file_path,
                                                        split=['train[:80%]', 'train[80%:]'])
    else:
        train_ds = datasets.load_dataset("json", data_files=sft_config.train_file_path)
        validation_ds = datasets.load_dataset("json", data_files=sft_config.validate_file_path)

    raw_datasets = datasets.DatasetDict({"train": train_ds, "validation": validation_ds})
    column_names = raw_datasets["train"].column_names if training_args.do_train else raw_datasets[
        "validation"].column_names

    def process_supervised(record):
        input_s = record['instruction'] + '\n' + (record['input'] or '')
        output_s = record['output']
        tokenized = tokenizer([input_s, output_s])
        token_ids = [tok_id for tok_ids in tokenized['input_ids'] for tok_id in tok_ids]
        attention_mask = [mask for masks in tokenized['attention_mask'] for mask in masks]
        processed_record = {
            "input_ids": token_ids[:sft_config.max_length],
            "attention_mask": attention_mask[:sft_config.max_length],
            "labels": token_ids.copy()[:sft_config.max_length]
        }
        # ignore input label, label is ignored if value is -100
        processed_record["labels"][:len(tokenized["input_ids"][0])] = [-100] * len(tokenized["input_ids"][0])
        return {k: torch.tensor(v, dtype=torch.int) for k, v in processed_record.items()}

    with training_args.main_process_first(desc="Process supervised dataset"):
        sft_dataset = raw_datasets.map(
            process_supervised,
            batched=False,
            num_proc=sft_config.preprocess_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            desc="Process supervised dataset"
        )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=sft_dataset["train"],
        eval_dataset=sft_dataset["validation"],
        tokenizer=tokenizer,  # trainer need tokenizer.pad_token_id,
        data_collator=transformers.DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest",
                                                                      max_length=sft_config.max_length,
                                                                      label_pad_token_id=-100),
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # trigger Training
    trainer.train()
    trainer.save_model()
    trainer.save_state()


if __name__ == '__main__':
    main()
"""
deepspeed \
--include="localhost:0,1,2,3" \
./train_sft.py \
--deepspeed ./ds_config/ds_config_zero3.json \
--model_path ./tigerbot_560m \
--do_train \
--train_file_path ./data/dev_sft.json \
--output_dir ./ckpt-sft \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 5 \
--learning_rate 1e-5 \
--evaluation_strategy steps \
--eval_steps 10 \
--bf16 True \
--save_strategy steps \
--save_steps 10 \
--save_total_limit 2 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2
"""
