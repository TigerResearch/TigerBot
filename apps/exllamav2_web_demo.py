import streamlit as st
import torch.cuda

from other_infer.exllamav2_hf_infer import get_model, generate_stream
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
generation_config.max_length = args.max_input_length + args.max_generate_length
generation_config.max_new_tokens = args.max_generate_length

tok_ins = "\n\n### Instruction:\n"
tok_res = "\n\n### Response:\n"
tok_eos = "</s>"

st.title("TigerBot chat")

device = f"cuda:{torch.cuda.current_device()}"

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
        for text in generate_stream(model, tokenizer, inputs['input_ids'], inputs['attention_mask'],
                                    generation_config=generation_config):
            print(text, end='', flush=True)
            full_answer += text
            message_placeholder.text(full_answer)
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
