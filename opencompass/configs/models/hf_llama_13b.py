from opencompass.models import HuggingFaceCausalLM


models = [
    # LLaMA 13B
    dict(
        type=HuggingFaceCausalLM,
        abbr='llama2-13b-hf',
        path="/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf",
        tokenizer_path='/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
