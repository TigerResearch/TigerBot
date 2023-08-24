from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TriviaQADataset, TriviaQAEvaluator

triviaqa_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='dev',
    test_split='dev')

triviaqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="Answer the final question, your answer should be as simple as possible.\nQ: Where in England was Dame Judi Dench born?\nA: York, England\nQ: From which country did Angola achieve independence in 1975?\nA: Portugal\nQ: What is Bruce Willis' real first name?\nA: Walter\nQ: Who won Super Bowl XX?\nA: Chicago Bears\nQ: How is Joan Molinsky better known?\nA: Joan Rivers\nQ: {question}?"),
                dict(role='BOT', prompt='A: '),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50))

triviaqa_eval_cfg = dict(
    evaluator=dict(type=TriviaQAEvaluator), pred_role='BOT')

triviaqa_datasets = [
    dict(
        type=TriviaQADataset,
        abbr='triviaqa',
        path='./data/triviaqa_wiki',
        reader_cfg=triviaqa_reader_cfg,
        infer_cfg=triviaqa_infer_cfg,
        eval_cfg=triviaqa_eval_cfg)
]
