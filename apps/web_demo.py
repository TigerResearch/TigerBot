import streamlit as st
import torch.cuda

from TigerBot.utils.modeling_hack import get_model
from TigerBot.utils.streaming import generate_stream


@st.cache_resource
def cached_get_model(model_path='/data1/cong.fu/models/hf/tigerbot-13b-chat-v4', rope_scaling='dynamic',
                     rope_factor=8.0):
    return get_model(model_path=model_path, rope_scaling=rope_scaling, rope_factor=rope_factor)


model, tokenizer, generation_config = cached_get_model()

generation_config.max_length = 16384
generation_config.max_new_tokens = 1024

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
        print(inputs)
        for text in generate_stream(model, tokenizer, inputs['input_ids'], inputs['attention_mask'],
                                    generation_config=generation_config):
            print(text, end='', flush=True)
            full_answer += text
            message_placeholder.text(full_answer)
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
