from mmengine.config import read_base

with read_base():
    from .models.hf_tigerbot_13b_base import models

    # 英文任务
    from .datasets.humaneval.humaneval_gen import humaneval_datasets
    from .datasets.piqa.piqa_ppl import piqa_datasets
    from .datasets.siqa.siqa_ppl import siqa_datasets
    from .datasets.hellaswag.hellaswag_ppl import hellaswag_datasets
    from .datasets.winogrande.winogrande_ppl import winogrande_datasets
    from .datasets.obqa.obqa_gen import obqa_datasets
    from .datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl import BoolQ_datasets
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from .datasets.mmlu.mmlu_ppl import mmlu_datasets

    #  中文任务
    from .datasets.CLUE_CMRC.CLUE_CMRC_gen import CMRC_datasets
    from .datasets.CLUE_ocnli.CLUE_ocnli_ppl import ocnli_datasets
    from .datasets.CLUE_C3.CLUE_C3_ppl import C3_datasets
    from .datasets.ceval.ceval_ppl import ceval_datasets

datasets1 = [*humaneval_datasets, *piqa_datasets, *siqa_datasets, *hellaswag_datasets, *winogrande_datasets,
             *obqa_datasets, *BoolQ_datasets, *gsm8k_datasets, *mmlu_datasets]
datasets2 = [*CMRC_datasets, *ocnli_datasets, *C3_datasets, *ceval_datasets]

datasets = datasets1 + datasets2
