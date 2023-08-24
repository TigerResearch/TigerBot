from opencompass.models import HuggingFace

models = [
    dict(
        type=HuggingFace,
        abbr='chatglm2-6b',
        path='THUDM/chatglm2-6b',
        tokenizer_path='THUDM/chatglm2-6b',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        batch_padding=True,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
