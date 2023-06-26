import json
import requests
import sseclient


def predict(prompt, history=None):
    if history is None:
        history = []
    session = requests.Session()
    url = 'http://localhost:8000/stream_chat'
    data = {"prompt": prompt, "history": history}
    headers = {'Content-Type': 'application/json'}
    event_source = sseclient.SSEClient(url, json=data, headers=headers, session=session)
    history = []
    for event in event_source:
        # 将事件传递给回调函数进行处理
        data = json.loads(event.data)
        if not data["finish"]:
            response = data["response"]
            history = data["history"]
            print("\rChatBot:" + response, end="")
        else:
            break
    print("")
    return history


if __name__ == "__main__":
    history1 = []
    while True:
        query = input("Human:")
        history = predict(query, history1)
