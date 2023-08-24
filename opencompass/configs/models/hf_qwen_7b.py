from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen-7B',
        path="Qwen/Qwen-7B",
        tokenizer_path='Qwen/Qwen-7B',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              trust_remote_code=True
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=4,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen-7b-chat',
        path="Qwen/Qwen-7B-Chat",
        tokenizer_path='Qwen/Qwen-7B-Chat',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              trust_remote_code=True
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=4,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
