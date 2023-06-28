from fastapi import FastAPI, Request
import uvicorn
import json
import datetime
import asyncio
import os
import sys
import torch
from sse_starlette.sse import EventSourceResponse
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from transformers import AutoTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from other_infer.infer_stream import get_model


os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


def get_prompt(query, history=None):
    if not history:
        prompt = "\n\n### Instruction:\n{}\n\n### Response:\n".format(query)
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "\n\n### Instruction:\n{}\n\n### Response:\n{}".format(
                old_query, response)
        prompt += "\n\n### Instruction:\n{}\n\n### Response:\n".format(
            query)
    return prompt


@app.get("/")
async def root():
    return "Hello! This is TigerBot API."


@app.post("/chat/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_input_length = json_post_list.get('max_input_length', 512)
    max_generate_length = json_post_list.get('max_generate_length', 1024)
    top_p = json_post_list.get('top_p', 0.95)
    temperature = json_post_list.get('temperature', 0.8)
    if tokenizer.model_max_length is None or tokenizer.model_max_length > max_generate_length:
        tokenizer.model_max_length = max_generate_length
    generation_kwargs = {
        "top_p": top_p,
        "temperature": temperature,
        "max_length": max_generate_length,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "early_stopping": True,
        "no_repeat_ngram_size": 4,
    }
    prompt = prompt.lstrip("\n")
    query = get_prompt(prompt, history)
    query = query.strip()
    inputs = tokenizer(query, return_tensors='pt', truncation=True,
                       max_length=max_input_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model.generate(**inputs, **generation_kwargs)
    response = ''
    for tok_id in output[0][inputs['input_ids'].shape[1]:]:
        if tok_id != tokenizer.eos_token_id:
            response += tokenizer.decode(tok_id)
    response = response.lstrip("\n").rstrip("\n### Response:")
    history += [(prompt, response)]
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    # log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    # print(log)
    torch_gc()
    return answer


@app.get("/stream_chat")
async def stream_chat(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_input_length = json_post_list.get('max_input_length', 512)
    max_generate_length = json_post_list.get('max_generate_length', 1024)
    top_p = json_post_list.get('top_p', 0.95)
    temperature = json_post_list.get('temperature', 0.8)
    if tokenizer.model_max_length is None or tokenizer.model_max_length > max_generate_length:
        tokenizer.model_max_length = max_generate_length
    STREAM_DELAY = 1  # second
    RETRY_TIMEOUT = 15000  # milisecond

    # def new_messages(prompt, history, max_length, top_p, temperature):
    #     response, history = 
    #     yield response, history
    
    async def event_generator(
            prompt, history, max_input_length, max_generate_length, top_p, temperature
    ):
        finished = False
        message_generator = model.stream_chat(
            tokenizer,
            prompt,
            history=history,
            max_input_length=max_input_length,
            max_generate_length=max_generate_length,
            top_p=top_p,
            temperature=temperature,
        )
        last_message = ["", ""]
        while not finished:
            # If client closes connection, stop sending events
            if await request.is_disconnected():
                break

            # Checks for new messages and return them to client if any
            try:
                message = next(message_generator)
                if message[0] is None:
                    finished = True
                    temp_dict = {
                        "response": last_message[0],
                        "history": last_message[1],
                        "finish": True
                    }
                    yield {
                        "event": "finish",
                        "id": "finish_id",
                        "retry": RETRY_TIMEOUT,
                        "data": json.dumps(temp_dict, ensure_ascii=False)
                    }
                    break
                else:
                    temp_dict = {
                        "response": message[0],
                        "history": message[1],
                        "finish": False
                    }
                    yield {
                        "event": "new_message",
                        "id": "message_id",
                        "retry": RETRY_TIMEOUT,
                        "data": json.dumps(temp_dict, ensure_ascii=False)
                    }
                    last_message = message
            except StopIteration:
                message_generator = model.stream_chat(
                    tokenizer,
                    prompt,
                    history=history,
                    max_input_length=max_input_length,
                    max_generate_length=max_generate_length,
                    top_p=top_p,
                    temperature=temperature,
                )
                await asyncio.sleep(STREAM_DELAY)
    torch_gc()
    return EventSourceResponse(
        event_generator(
            prompt,
            history,
            max_input_length,
            max_generate_length,
            top_p,
            temperature
        )
    )


if __name__ == '__main__':
    model_path = "tigerbot-7b-sft"
    model_max_length = 1024
    print(f"loading model: {model_path}...")
    model = get_model(model_path)
    max_memory = get_balanced_memory(model)
    device_map = infer_auto_device_map(model, max_memory=max_memory,
                                       no_split_module_classes=["BloomBlock"])
    print("Using the following device map for the model:", device_map)
    model = dispatch_model(model, device_map=device_map, offload_buffers=True)

    device = torch.cuda.current_device()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="left",
        truncation_side='left',
        padding=True,
        truncation=True
    )
    device = torch.device(CUDA_DEVICE)
    model = model.to(device)
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
