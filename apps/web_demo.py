import torch
import os
import sys
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, GenerationConfig
import mdtex2html
from threading import Thread
import gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
max_generate_length: int = 1024
model_path = "TigerBot"
print(f"loading model: {model_path}...")
device = torch.cuda.current_device()
generation_config = GenerationConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    cache_dir=None,
    model_max_length=max_generate_length,
    padding_side="left",
    truncation_side='left',
    padding=True,
    truncation=True
)
if tokenizer.model_max_length is None or tokenizer.model_max_length > 1024:
    tokenizer.model_max_length = 1024

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


def generate_stream(query,
                    history,
                    max_input_length,
                    max_output_length):
    tok_ins = "\n\n### Instruction:\n"
    tok_res = "\n\n### Response:\n"
    prompt_input = tok_ins + "{instruction}" + tok_res

    sess_text = ""
    if history:
        for s in history:
            sess_text += tok_ins + s["human"] + tok_res + s["assistant"]
    history.append({"human": query, "assistant": ""})

    sess_text += tok_ins + query.strip()
    input_text = prompt_input.format_map({'instruction': sess_text.split(tok_ins, 1)[1]})
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer,
                                    skip_prompt=True,
                                    skip_special_tokens=True,
                                    spaces_between_special_tokens=False)

    generation_kwargs = generation_config.to_dict()
    generation_kwargs.update(dict(inputs))
    generation_kwargs['streamer'] = streamer
    generation_kwargs['max_new_tokens'] = max_output_length

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    answer = ""
    for new_text in streamer:
        if len(new_text) == 0:
            continue
        if new_text.endswith(tokenizer.eos_token):
            new_text = new_text.rsplit(tokenizer.eos_token, 1)[0]
        answer += new_text
        history[-1]['assistant'] = answer

        yield answer, history


def predict(input, chatbot, max_input_length, max_generate_length, history):
    chatbot.append((parse_text(input), ""))
    for response, history in generate_stream(
            input,
            history,
            max_input_length=max_input_length,
            max_output_length=max_generate_length,
    ):
        if response is None:
            break
        chatbot[-1] = (parse_text(input), parse_text(response))

        yield chatbot, history


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

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_input_length, max_generate_length, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=True, inbrowser=True)
