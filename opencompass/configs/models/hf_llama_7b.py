from opencompass.models import HuggingFaceCausalLM

models = [
    # LLaMA 7B
    dict(
        type=HuggingFaceCausalLM,
        abbr='llama-7b',
        path="meta-llama/Llama-2-7b-hf",
        tokenizer_path='meta-llama/Llama-2-7b',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto'),
        batch_padding=True,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        type=HuggingFaceCausalLM,
        abbr='llama-7b-chat',
        path="meta-llama/Llama-2-7b-chat-hf",
        tokenizer_path='meta-llama/Llama-2-7b-chat-hf',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto'),
        batch_padding=True,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]
