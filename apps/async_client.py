import asyncio
import json

import aiohttp_sse_client.client
from aiohttp import ClientSession
from aiohttp_sse_client import client as sseclient


async def handle_event(event: aiohttp_sse_client.client.MessageEvent, event_source):
    # 处理 SSE 事件的回调函数
    # print(f'Event type: {event.type}')
    # print(f'Event id: {event.last_event_id}')
    # print(f'Event data: {event.data}')
    # print(f'Event message: {event.message}')

    data = json.loads(event.data)
    # print("data", data)
    if event.type == "finish":
        try:
            await event_source.close()
        except Exception as err:
            print("close with error", err)
    return data["response"], data["history"], event.type


async def listen_sse(prompt, history=None, max_length=2048, top_p=0.7, temperature=0.96):
    if history is None:
        history = []
    async with ClientSession() as session:
        url = 'http://127.0.0.1:8000/stream_chat'
        data = {
            "prompt": prompt,
            "history": history,
            "max_length": max_length,
            "top_p": top_p,
            "temperature": temperature,
        }
        headers = {'Content-Type': 'application/json'}
        response, history = None, None
        print("=" * 100)
        async with sseclient.EventSource(url, json=data, headers=headers, session=session) as event_source:
            try:
                async for event in event_source:
                    # 将事件传递给回调函数进行处理
                    response, history, e_type = await handle_event(event, event_source)
                    print(response, end="", flush=True)
                    if e_type == "finish":
                        break
            except Exception as err:
                print("event close", err)
        print()
        print("=" * 100)
        return response, history


if __name__ == "__main__":
    history = []
    while True:
        raw_text = input(
            "prompt(\"exit\" to end, \"clear\" to clear session) >>> ")
        if not raw_text:
            print('prompt should not be empty!')
            continue
        if raw_text.strip() == "exit":
            print('session ended.')
            break
        if raw_text.strip() == "clear":
            print('session cleared.')
            history = []
            continue
        query_text = raw_text.strip()
        _, history1 = asyncio.run(listen_sse(query_text, history))
