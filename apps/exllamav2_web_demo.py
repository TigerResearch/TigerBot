import streamlit as st
import torch.cuda

from other_infer.exllamav2_hf_infer import get_model
from utils.streaming import generate_stream
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--max_input_length', type=int, required=False, default=1024)
parser.add_argument('--max_generate_length', type=int, required=False, default=1024)

args = parser.parse_args()
print(args)


@st.cache_resource
def cached_get_model(model_path):
    print(f'Init model: {model_path}')
    return get_model(model_path=model_path)


model, tokenizer, generation_config = cached_get_model(args.model_path)
generation_config.do_sample = False
generation_config.max_length = args.max_input_length + args.max_generate_length
generation_config.max_new_tokens = args.max_generate_length


tok_ins = "\n\n### Instruction:\n"
tok_res = "\n\n### Response:\n"
tok_eos = "</s>"

st.title("TigerBot chat")

device = f"cuda:{torch.cuda.current_device()}"


def main(
        model_path: str,
        max_input_length: int = 512,
        max_generate_length: int = 2048,
        stream: bool = True,
):
    print(f"loading model: {model_path}...")

    model = get_model(model_path)
    device = torch.cuda.current_device()
    model, tokenizer, generation_config = get_model(model_path)

    generation_config.do_sample = False
    generation_config.max_length = max_input_length + max_generate_length
    generation_config.max_new_tokens = max_generate_length

    sess_text = ""

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
    )
    generation_kwargs = generation_config.to_dict()

    def eval_generate(**args):
        with torch.inference_mode(mode=True):
            model.eval()
            model.generate(**args)

    with torch.inference_mode(mode=True):
        model.eval()
        while True:
            raw_text = input(
                'prompt("exit" to end, "clear" to clear session) >>> '
            )
            if not raw_text:
                print("prompt should not be empty!")
                continue
            if raw_text.strip() == "exit":
                print("session ended.")
                break
            if raw_text.strip() == "clear":
                print("session cleared.")
                sess_text = ""
                continue

            query_text = raw_text.strip()
            sess_text += tok_ins + query_text
            input_text = prompt_input.format_map(
                {"instruction": sess_text.split(tok_ins, 1)[1]}
            )
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length,
            )
            tic = time.perf_counter()
            if stream:
                generation_kwargs["streamer"] = streamer
                for k, v in inputs.items():
                    generation_kwargs[k] = v.to(device)
                thread = Thread(
                    target=eval_generate, kwargs=generation_kwargs
                )
                thread.start()
                answer = ""
                flag = False
                print("=" * 100)
                for new_text in streamer:
                    if new_text.endswith(tokenizer.eos_token):
                        new_text = new_text.rsplit(
                            tokenizer.eos_token, 1
                        )[0].strip()
                        flag = True
                    print(new_text, end="")
                    answer += new_text
                    if flag:
                        break
                toc = time.perf_counter()
                num_tok = len(tokenizer.encode(answer))
            else:
                if "streamer" in generation_kwargs:
                    del generation_kwargs["streamer"]
                inputs = {k: v.to(device) for k, v in inputs.items()}
                output = model.generate(
                    **inputs, **generation_config.to_dict()
                )
                toc = time.perf_counter()
                num_tok = output.shape[1]
                output_str = tokenizer.decode(
                    output[0],
                    skip_special_tokens=False,
                    spaces_between_special_tokens=False,
                )
                answer = output_str.rsplit(tok_res, 1)[1].strip()
                if answer.endswith(tokenizer.eos_token):
                    answer = answer.rsplit(tokenizer.eos_token, 1)[
                        0
                    ].strip()
                print(answer)

            sess_text += tok_res + answer
            res_time = toc - tic
            print(
                f"\n[time: {res_time:0.4f} sec, speed: {num_tok / res_time:0.4f} tok/sec]"
            )
            print("=" * 100)

if "messages" not in st.session_state:
    st.session_state.messages = list()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.text(message["content"])

if prompt := st.chat_input("Input here"):
    with st.chat_message("user"):
        st.text(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        input_text = "".join(
            [(tok_ins if msg["role"] == "user" else tok_res) + msg["content"] + (
                tok_eos if msg["role"] == "assistant" else "") for msg in st.session_state.messages])
        input_text += tok_res
        inputs = tokenizer(input_text, return_tensors='pt',
                           max_length=generation_config.max_length - generation_config.max_new_tokens)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        full_answer = ""
        print(inputs)
        for text in generate_stream(model, tokenizer, inputs['input_ids'], inputs['attention_mask'],
                                    generation_config=generation_config):
            print(text, end='', flush=True)
            full_answer += text
            message_placeholder.text(full_answer)
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
