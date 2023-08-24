from mmengine.config import read_base

with read_base():
    # 英文任务
    from .datasets.humaneval.humaneval_gen import humaneval_datasets
    from .datasets.hellaswag.hellaswag_ppl import hellaswag_datasets
    from .datasets.winogrande.winogrande_ppl import winogrande_datasets
    from .datasets.obqa.obqa_gen import obqa_datasets

datasets = [*humaneval_datasets, *hellaswag_datasets, *winogrande_datasets, *obqa_datasets]

from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='tigerbot-7b-chat-1',
        path="TigerResearch/tigerbot-7b-chat",
        tokenizer_path='TigerResearch/tigerbot-7b-chat',
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
