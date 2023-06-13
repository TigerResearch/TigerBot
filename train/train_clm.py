import os
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import transformers

os.environ["WANDB_DISABLED"] = "true"


@dataclass
class PretrainConfig:
    model_name_or_path: Optional[str] = field(metadata={"help": "Path to pretrained model checkpoint"})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Huggingface dataset name"})
    train_file_path: Optional[str] = field(default=None, metadata={"help": "Path to train data file/directory"})
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
    parser = transformers.HfArgumentParser((PretrainConfig, transformers.TrainingArguments))
    pretrain_config, training_args = parser.parse_args_into_dataclasses()

    # check file existence
    if pretrain_config.dataset_name is None and pretrain_config.train_file_path is None:
        raise ValueError(f"One of --dataset_name or --train_file_path must be set")
    if pretrain_config.train_file_path:
        check_file_exist(pretrain_config.train_file_path)
    if pretrain_config.validate_file_path:
        check_file_exist(pretrain_config.validate_file_path)

    # load model, tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrain_config.model_name_or_path, padding_side='right',
                                                           trunction_side="right",
                                                           max_length=pretrain_config.max_length)

    model = transformers.AutoModelForCausalLM.from_pretrained(pretrain_config.model_name_or_path)

    if pretrain_config.dataset_name:
        ds = datasets.load_dataset(pretrain_config.dataset_name)
        train_ds, validation_ds = ds['train'], ds['validation']
        raw_datasets = datasets.DatasetDict({"train": train_ds, "validation": validation_ds})
    else:
        # Split 20% of train data as validation data
        if not pretrain_config.validate_file_path:
            train_ds, validation_ds = datasets.load_dataset('json', data_files=pretrain_config.train_file_path,
                                                            split=['train[:80%]', 'train[80%:]'])
            raw_datasets = datasets.DatasetDict({"train": train_ds, "validation": validation_ds})
        else:
            raw_datasets = datasets.load_dataset("json", data_files={'train': pretrain_config.train_file_path,
                                                                     'validation': pretrain_config.validate_file_path})

    column_names = raw_datasets["train"].column_names if training_args.do_train else raw_datasets[
        "validation"].column_names

    def process_pretrain(batch):
        texts = batch[pretrain_config.text_key_name]
        texts = [text if text.endswith(tokenizer.eos_token) else text + tokenizer.eos_token for text in texts]
        tokenized = tokenizer(texts)
        for k in tokenized.keys():
            batch[k] = [e[:pretrain_config.max_length] for e in tokenized[k]]
        batch['labels'] = batch['input_ids'].copy()
        return batch

    with training_args.main_process_first(desc="Process pretrain dataset"):
        clm_dataset = raw_datasets.map(
            process_pretrain,
            batched=True,
            batch_size=4,
            num_proc=pretrain_config.preprocess_num_workers,
            remove_columns=column_names,
            desc="Process pretrain dataset"
        )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=clm_dataset["train"],
        eval_dataset=clm_dataset["validation"],
        tokenizer=tokenizer,  # trainer need tokenizer.pad_token_id,
        data_collator=transformers.DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest",
                                                                      max_length=pretrain_config.max_length,
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
./train_clm.py \
--deepspeed ./ds_config/ds_config_zero3.json \
--model_name_or_path TigerResearch/tigerbot-7b-base \
--dataset_name TigerResearch/dev_pretrain \
--do_train \
--output_dir ./ckpt-clm \
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
