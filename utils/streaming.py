import threading
from typing import Iterator

import torch
import transformers
import string


def put(self, value):
    """
    Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
    """
    if len(value.shape) > 1 and value.shape[0] > 1:
        raise ValueError("TextStreamer only supports batch size 1")
    elif len(value.shape) > 1:
        value = value[0]

    if self.skip_prompt and self.next_tokens_are_prompt:
        self.next_tokens_are_prompt = False
        return

    # Add the new token to the cache and decodes the entire thing.
    self.token_cache.extend(value.tolist())
    text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n"):
        printable_text = text[self.print_len:]
        self.token_cache = []
        self.print_len = 0
    # If the last token is a CJK character, we print the characters.
    elif len(text) > 0 and (
            self._is_chinese_char(ord(text[-1])) or text[-1] in string.ascii_letters or text[-1] in set(
        '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~！“”#%&、‘’（）*+，-。/：；《=》？@【】·「｜」 \t\n\r\x0b\x0c')):
        printable_text = text[self.print_len:]
        self.print_len += len(printable_text)
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
        printable_text = text[self.print_len: text.rfind(" ") + 1]
        self.print_len += len(printable_text)

    self.on_finalized_text(printable_text)


transformers.TextIteratorStreamer.put = put


def generate_stream(model: transformers.AutoModelForCausalLM, tokenizer: transformers.AutoModelForCausalLM,
                    input_ids: torch.Tensor, attention_mask: torch.Tensor,
                    generation_config: transformers.GenerationConfig):
    streamer = transformers.TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=180)
    kwargs = generation_config.to_dict()
    kwargs['input_ids'] = input_ids
    kwargs['attention_mask'] = attention_mask
    kwargs['streamer'] = streamer
    threading.Thread(target=model.generate, kwargs=kwargs).start()

    return streamer
