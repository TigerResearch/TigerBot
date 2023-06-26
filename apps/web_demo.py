import torch
import os
import gradio as gr
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from transformers import AutoTokenizer
import mdtex2html

def get_model(model):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    return model

os.environ["TOKENIZERS_PARALLELISM"] = "false"
max_generate_length: int = 1024
model_path = "tigerbot-7b-sft"
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


def predict(input, chatbot, max_input_length, max_generate_length, top_p, temperature, history):
    chatbot.append((parse_text(input), ""))
    for response, history in model.stream_chat(
            tokenizer,
            input,
            history,
            max_input_length=max_input_length,
            max_generate_length=max_generate_length,
            top_p=top_p,
            temperature=temperature
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
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_input_length, max_generate_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=False, inbrowser=True)
