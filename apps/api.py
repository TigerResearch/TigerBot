from fastapi import FastAPI, Request
import uvicorn
import json
import datetime
import asyncio
import os
import sys
import time
import torch
from threading import Thread
from sse_starlette.sse import EventSourceResponse
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    device = torch.cuda.current_device()
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
    # log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' +
    repr(response) + '"'  # print(log)
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

    async def event_generator(prompt, history, max_input_length, max_generate_length, top_p, temperature):
        last_message = ["", ""]
        streamer = TextIteratorStreamer(tokenizer,
                                        skip_prompt=True,
                                        skip_special_tokens=True,
                                        spaces_between_special_tokens=False)

        query = get_prompt(prompt, history)
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=max_input_length)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        generation_kwargs = {
            **inputs,
            "top_p": top_p,
            "streamer": streamer,
            "temperature": temperature,
            "max_new_tokens": max_generate_length,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "early_stopping": True,
            "no_repeat_ngram_size": 4,
        }
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        last_msg = ""
        for new_text in streamer:
            # If client closes connection, stop sending events
            if await request.is_disconnected():
                break
            # Checks for new messages and return them to client if any
            try:
                temp_dict = {
                    "response": new_text,
                    "history": history + [(prompt, last_msg)],
                    "finish": False
                }
                yield {
                    "event": "new_message",
                    "id": "message_id",
                    "retry": RETRY_TIMEOUT,
                    "data": json.dumps(temp_dict, ensure_ascii=False)
                }
                last_msg += new_text
            except StopIteration:
                await asyncio.sleep(STREAM_DELAY)
        temp_dict = {
            "response": new_text,
            "history": history + [(prompt, last_msg)],
            "finish": True
        }
        yield {
            "event": "finish",
            "id": "finish_id",
            "retry": RETRY_TIMEOUT,
            "data": json.dumps(temp_dict, ensure_ascii=False)
        }
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


# --- Compatible with OpenAI ChatGPT --- #
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice,
                        ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda:
    int(time.time()))


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        query = prev_messages.pop(0).content + query

    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and prev_messages[i +
                                                                 1].role == "assistant":
                history.append([prev_messages[i].content, prev_messages[i
                                                                        + 1].content])

    if request.stream:
        generate = predict(query, history, request.model)
        return EventSourceResponse(generate,
                                   media_type="text/event-stream")

    response, _ = model.chat(tokenizer, query, history=history)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model,
                                  choices=[choice_data], object="chat.completion")


async def predict(query: str, history: List[List[str]], model_id: str):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data],
                                   object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    current_length = 0

    for new_response, _ in model.stream_chat(tokenizer, query, history):
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    yield '[DONE]'


if __name__ == '__main__':
    model_path = "/mnt/nfs/yechen/models/tigerbot-7b-2h-sft-20g-mix0.0-group-mg-hf-9600"
    model_max_length = 1024
    print(f"loading model: {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="left",
        truncation_side='left',
        padding=True,
        truncation=True
    )
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
