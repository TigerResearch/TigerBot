from mmengine.config import read_base

with read_base():
    # 英文任务
    from .datasets.piqa.piqa_ppl import piqa_datasets
    from .datasets.siqa.siqa_ppl import siqa_datasets
    from .datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl import BoolQ_datasets
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from .datasets.mmlu.mmlu_ppl import mmlu_datasets

    #  中文任务
    from .datasets.CLUE_CMRC.CLUE_CMRC_gen_chat import CMRC_datasets
    from .datasets.CLUE_ocnli.CLUE_ocnli_ppl import ocnli_datasets
    from .datasets.CLUE_C3.CLUE_C3_ppl import C3_datasets
    from .datasets.ceval.ceval_ppl import ceval_datasets

datasets = [*piqa_datasets, *siqa_datasets, *BoolQ_datasets, *gsm8k_datasets, *mmlu_datasets, *CMRC_datasets,
            *ocnli_datasets, *C3_datasets, *ceval_datasets]

from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='\n\n### Instruction:\n'),
        dict(role='BOT', begin='\n\n### Response:\n', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='tigerbot-7b-2',
        path="TigerResearch/tigerbot-7b-chat",
        tokenizer_path='TigerResearch/tigerbot-7b-chat',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_out_len=200,
        max_seq_len=2048,
        batch_size=16,
        meta_template=_meta_template,
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        batch_padding=True,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
