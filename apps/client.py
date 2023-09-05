import json
import requests
import sseclient


def predict(prompt, history=None):
    if history is None:
        history = []
    session = requests.Session()
    url = 'http://127.0.0.1:8000/stream_chat'
    data = {"prompt": prompt, "history": history}
    headers = {'Content-Type': 'application/json'}
    event_source = sseclient.SSEClient(url, json=data, headers=headers, session=session)
    history = []
    print("=" * 100)
    for event in event_source:
        # 将事件传递给回调函数进行处理
        try:
            data = json.loads(event.data)
            if not data["finish"]:
                response = data["response"]
                history = data["history"]
                print(response, end="", flush=True)
            else:
                break
        except:
            break
    print()
    print("=" * 100)
    return history


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
        history = predict(query_text, history)
