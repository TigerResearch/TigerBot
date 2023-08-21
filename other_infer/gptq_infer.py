import time
import os
import fire
import random
import torch
import logging
from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class QuantAutoGPTQ:
    def __init__(self, model_path, output_dir=None, dataset=None, model_basename=None,
                 num_samples=128, trust_remote_code=False, cache_examples=True,
                 use_fast=True, use_triton=False, bits=[4], group_size=[128], damp=[0.01],
                 desc_act=[0], dtype='float16', seqlen=2048, batch_size=1, max_input_length=512, max_generate_length=1024):

        self.model_path = model_path
        self.output_dir_base = output_dir
        self.dataset = dataset
        self.num_samples = num_samples
        self.trust_remote_code = trust_remote_code
        self.cache_examples = cache_examples
        self.use_fast = use_fast
        self.use_triton = use_triton
        self.model_basename = model_basename
        self.max_input_length = max_input_length
        self.max_generate_length = max_generate_length

        quant = (output_dir is not None)

        if quant:
            assert output_dir is not None and dataset is not None, 'output_dir and dataset cannot be None when quantizing the model'
        else:
            assert model_basename is not None, 'model_basename cannot be None when inferring the quantized model'

        def check_list(item):
            return item if isinstance(item, list) else [item]

        self.bits = check_list(bits)
        self.group_size = check_list(group_size)
        self.desc_act = check_list(desc_act)
        self.damp = check_list(damp)

        self.dtype = dtype
        self.seqlen = seqlen
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)
        self.quant = quant

        self.logger.info("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                       use_fast=self.use_fast,
                                                       trust_remote_code=self.trust_remote_code)

        if self.quant:
            self.run_quantization()
        else:
            self.infer()

    def get_wikitext2(self):
        wikidata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        wikilist = [' \n' if s == '' else s for s in wikidata['text']]

        text = ''.join(wikilist)
        self.logger.info("Tokenising wikitext2")
        tokenized = self.tokenizer(text, return_tensors='pt')

        return self.append_dataset(tokenized, self.num_samples, self.seqlen)

    def get_ptb(self):
        traindata = load_dataset(
            'ptb_text_only', 'penn_treebank', split='train')

        self.logger.info("Tokenising ptb")
        tokenized = self.tokenizer("\n\n".join(
            traindata['sentence']), return_tensors='pt')

        return self.append_dataset(tokenized, self.num_samples, self.seqlen)

    def get_c4(self):
        traindata = load_dataset(
            'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train',  token=False
        )
        trainloader = []
        for _ in range(self.num_samples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = self.tokenizer(
                    traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= self.seqlen:
                    break
            i = random.randint(
                0, trainenc.input_ids.shape[1] - self.seqlen - 1)
            j = i + self.seqlen
            inp = trainenc.input_ids[:, i:j]
            attention_mask = torch.ones_like(inp)
            trainloader.append(
                {'input_ids': inp, 'attention_mask': attention_mask})
        return trainloader

    @staticmethod
    def append_dataset(tokenized, num_samples, seqlen):
        import numpy as np
        import torch

        random.seed(0)
        np.random.seed(0)
        torch.random.manual_seed(0)

        traindataset = []
        for _ in range(num_samples):
            i = random.randint(0, tokenized.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = tokenized.input_ids[:, i:j]
            attention_mask = torch.ones_like(inp)
            traindataset.append(
                {'input_ids': inp, 'attention_mask': attention_mask})
        return traindataset

    def quantize(self, output_dir, traindataset, bits, group_size, desc_act, damp):
        os.environ['BITSANDBYTES_NOWELCOME'] = '1'

        import torch
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            damp_percent=damp
        )

        if self.dtype == 'float16':
            torch_dtype = torch.float16
        elif self.dtype == 'float32':
            torch_dtype = torch.float32
        elif self.dtype == 'bfloat16':
            torch_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        self.logger.info(
            f"Loading model from {self.model_path} with trust_remote_code={self.trust_remote_code} and dtype={torch_dtype}")
        model = AutoGPTQForCausalLM.from_pretrained(self.model_path, quantize_config=quantize_config,
                                                    low_cpu_mem_usage=True, torch_dtype=torch_dtype, trust_remote_code=self.trust_remote_code)

        self.logger.info(
            f"Starting quantization to {output_dir} with use_triton={self.use_triton}")
        start_time = time.time()
        model.quantize(traindataset, use_triton=self.use_triton,
                       batch_size=self.batch_size, cache_examples_on_gpu=self.cache_examples)

        self.logger.info(
            f"Time to quantize model at {output_dir} with use_triton={self.use_triton}: {time.time() - start_time:.2f}")

        self.logger.info(f"Saving quantized model to {output_dir}")
        model.save_quantized(output_dir, use_safetensors=True)
        self.logger.info("Done.")

    def run_quantization(self):
        # TODO: This is messy, should be dynamic
        if 'wikitext2' in self.dataset:
            traindataset = self.get_wikitext2()
        elif 'ptb' in self.dataset:
            traindataset = self.get_ptb()
        elif 'c4' in self.dataset:
            traindataset = self.get_c4()
        else:
            self.logger.error(f"Unsupported dataset: {self.dataset}")
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        iterations = []
        for bits in self.bits:
            for group_size in self.group_size:
                for desc_act in self.desc_act:
                    for damp in self.damp:
                        desc_act = desc_act == 1 and True or False
                        iterations.append(
                            {"bits": bits, "group_size": group_size, "desc_act": desc_act, "damp": damp})

        num_iters = len(iterations)
        if num_iters > 1:
            self.logger.info(f"Starting {num_iters} quantizations.")
        count = 1
        for iteration in iterations:
            bits = iteration['bits']
            group_size = iteration['group_size']
            desc_act = iteration['desc_act']
            damp = iteration['damp']

        try:
            output_dir = os.path.join(
                self.output_dir_base, f"{bits}bits-{group_size}g-desc_act_{desc_act}-damp_{damp}")
            os.makedirs(output_dir, exist_ok=True)

            try:
                if num_iters > 1:
                    self.logger.info(
                        f"Starting quantization {count}/{num_iters}")
                self.logger.info(
                    f"Quantising with bits={bits} group_size={group_size} desc_act={desc_act} damp={damp} to {output_dir}")
                self.quantize(output_dir, traindataset, bits,
                              group_size, desc_act, damp)
            except KeyboardInterrupt:
                self.logger.error(f"Aborted. Will delete {output_dir}")
                os.rmdir(output_dir)
            except:
                raise

        finally:
            count += 1

    def get_model(self):
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        # from accelerate import infer_auto_device_map, dispatch_model
        # from accelerate.utils import get_balanced_memory

        def skip(*args, **kwargs):
            pass
        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        quantize_config = BaseQuantizeConfig(
            bits=self.bits[0],
            group_size=self.group_size[0],
            desc_act=self.desc_act[0],
            damp_percent=self.damp[0],
            static_groups=False,
            sym=True,
            true_sequential=True
        )

        model = AutoGPTQForCausalLM.from_quantized(self.model_path,
                                                   model_basename=self.model_basename,
                                                   use_safetensors=True,
                                                   trust_remote_code=True,
                                                   device_map='auto',
                                                   use_triton=self.use_triton,
                                                   quantize_config=quantize_config)

        # DONOT SUPPORT ACCELERATE !!!
        # max_memory = get_balanced_memory(model)
        # device_map = infer_auto_device_map(model, max_memory=max_memory,
        #                                 no_split_module_classes=[])
        # print("Using the following device map for the model:", device_map)
        # model = dispatch_model(model, device_map=device_map, offload_buffers=True)
        return model

    def infer(self):
        device = torch.cuda.current_device()
        tok_ins = "\n\n### Instruction:\n"
        tok_res = "\n\n### Response:\n"
        prompt_input = tok_ins + "{instruction}" + tok_res

        if self.tokenizer.model_max_length is None or self.tokenizer.model_max_length > self.max_generate_length:
            self.tokenizer.model_max_length = self.max_generate_length

        model = self.get_model()

        generation_kwargs = {
            "top_p": 0.95,
            "temperature": 0.8,
            "max_length": self.max_generate_length,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "early_stopping": True,
            "no_repeat_ngram_size": 4,
        }

        sess_text = ""
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
                sess_text = ""
                continue

            query_text = raw_text.strip()
            sess_text += tok_ins + query_text
            input_text = prompt_input.format_map(
                {'instruction': sess_text.split(tok_ins, 1)[1]})
            inputs = self.tokenizer(
                input_text, return_tensors='pt', truncation=True, max_length=self.max_input_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model.generate(**inputs, **generation_kwargs)
            output_str = self.tokenizer.decode(
                output[0], skip_special_tokens=False, spaces_between_special_tokens=False)
            answer = output_str.rsplit(tok_res, 1)[1].strip()
            if answer.endswith(self.tokenizer.eos_token):
                answer = answer.rsplit(self.tokenizer.eos_token, 1)[0].strip()
            sess_text += tok_res + answer

            print("=" * 100)
            print(answer)
            print("=" * 100)


if __name__ == "__main__":
    fire.Fire(QuantAutoGPTQ)
