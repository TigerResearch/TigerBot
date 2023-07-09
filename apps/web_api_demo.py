import os
import gradio as gr
import mdtex2html
import sseclient
import requests
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
max_generate_length: int = 1024
"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_input_length, max_generate_length, top_p, temperature, history):
    if history is None:
        history = []
    chatbot.append((parse_text(input), ""))
    session = requests.Session()
    url = 'http://localhost:8000/stream_chat'
    data = {
        "prompt": input,
        "history": history,
        "max_input_length": max_input_length,
        "max_generate_length": max_generate_length,
        "top_p": top_p,
        "temperature": temperature,
    }
    headers = {'Content-Type': 'application/json'}
    event_source = sseclient.SSEClient(url, json=data, headers=headers,
                                       session=session)
    for event in event_source:
        # 将事件传递给回调函数进行处理
        data = json.loads(event.data)
        if not data["finish"]:
            response = data["response"]
            history = data["history"]
            chatbot[-1] = (parse_text(input), parse_text(response))
            yield chatbot, history
        else:
            break


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">TigerBot</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_input_length = gr.Slider(0, 1024, value=512, step=1.0, label="Maximum input length", interactive=True)
            max_generate_length = gr.Slider(0, 2048, value=1024, step=1.0, label="Maximum generate length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_input_length, max_generate_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=False, inbrowser=False, server_name="0.0.0.0")
