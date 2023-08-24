from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='\n\n### Instruction:\n:'),
        dict(role='BOT', begin='\n\n### Response:\n:', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='tigerbot-7b-2h-sft-20g-mix0.0-group-mg-hf-9600',
        path="/mnt/nfs/yechen/models/tigerbot-7b-2h-sft-20g-mix0.0-group-mg-hf-9600",
        tokenizer_path='/mnt/nfs/yechen/models/tigerbot-7b-2h-sft-20g-mix0.0-group-mg-hf-9600',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        meta_template=_meta_template,
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        batch_padding=True,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
