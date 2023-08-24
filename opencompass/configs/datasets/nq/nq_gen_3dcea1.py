from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import NaturalQuestionDataset, NQEvaluator

nq_reader_cfg = dict(
    input_columns=['question'], output_column='answer', train_split='test')

nq_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt="Question: who sings does he love me with reba?"
                ),
                dict(
                    role="BOT",
                    prompt="Answer: Linda Davis\n"
                ),
                dict(
                    role="HUMAN",
                    prompt="Question: how many pages is invisible man by ralph ellison?"
                ),
                dict(
                    role="BOT",
                    prompt="Answer: 581 ( second edition )\n"
                ),
                dict(
                    role="HUMAN",
                    prompt="Question: where do the great lakes meet the ocean?"
                ),
                dict(
                    role="BOT",
                    prompt="Answer: the Saint Lawrence River\n"
                ),
                dict(
                    role="HUMAN",
                    prompt="Question: when does the new my hero academia movie come out?"
                ),
                dict(
                    role="BOT",
                    prompt="Answer: July 5 , 2018\n"
                ),
                dict(
                    role="HUMAN",
                    prompt="Question: what is the smallest prime number that is greater than 30?"
                ),
                dict(
                    role="BOT",
                    prompt="Answer: 31\n"
                ),
                dict(role='HUMAN', prompt='Question: {question}?\nAnswer: '),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

nq_eval_cfg = dict(evaluator=dict(type=NQEvaluator), pred_role="BOT")

nq_datasets = [
    dict(
        type=NaturalQuestionDataset,
        abbr='nq',
        path='./data/nq/',
        reader_cfg=nq_reader_cfg,
        infer_cfg=nq_infer_cfg,
        eval_cfg=nq_eval_cfg)
]
