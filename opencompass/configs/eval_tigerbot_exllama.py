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

from opencompass.models import ExllamaCausalLM

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='\n\n### Instruction:\n'),
        dict(role='BOT', begin='\n\n### Response:\n', generate=True),
    ],
)

models = [
    dict(
        type= ExllamaCausalLM,
        abbr='tigerbot',
        path="/mnt/nfs/algo/intern/yuwang/Tigerbot_AutoGPTQ/tigerbot_13b/tigerbot_13b_chat_4bit_c4_128g_no_act",
        tokenizer_path='/mnt/nfs/algo/intern/yuwang/Tigerbot_AutoGPTQ/tigerbot_13b/tigerbot_13b_chat_4bit_c4_128g_no_act',
        tokenizer_kwargs=dict(
            cache_dir=None,
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            padding=True,
            truncation=True,
            add_bos_token=False
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=4,
        batch_padding=True,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
