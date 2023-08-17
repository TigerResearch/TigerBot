import os
import bz2
import ctypes
import base64
import fire
import torch
import readline
from typing import List
from torch.nn import Linear
from torch.nn.parameter import Parameter
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tok_ins = "\n\n### Instruction:\n"
tok_res = "\n\n### Response:\n"
prompt_input = tok_ins + "{instruction}" + tok_res

logger = logging.get_logger(__name__)
RESOURCE_PACKAGE_NAME = __name__

try:
    from cpm_kernels.kernels.base import LazyKernelCModule, KernelFunction, round_up


    class Kernel:
        def __init__(self, code: bytes, function_names: List[str]):
            self.code = code
            self._function_names = function_names
            self._cmodule = LazyKernelCModule(self.code)

            for name in self._function_names:
                setattr(self, name, KernelFunction(self._cmodule, name))


    quantization_code = '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'

    kernels = Kernel(
        bz2.decompress(base64.b64decode(quantization_code)),
        [
            "int4WeightCompression",
            "int4WeightExtractionFloat",
            "int4WeightExtractionHalf",
            "int8WeightExtractionFloat",
            "int8WeightExtractionHalf",
        ],
    )
except Exception as exception:
    kernels = None
    logger.warning("Failed to load cpm_kernels:" + str(exception))


def get_model(model, wbit):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model = quantize(model, wbit)
    return model


def main(
        model_path: str,
        max_input_length: int = 512,
        max_generate_length: int = 1024,
        wbit: int = 8,
):
    assert wbit in [4, 8], 'weight_bit_width can only be 4 or 8!'

    print(f"loading model: {model_path}...")

    model = get_model(model_path, wbit)
    max_memory = get_balanced_memory(model)
    device_map = infer_auto_device_map(model, max_memory=max_memory,
                                       no_split_module_classes=[])
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
    if tokenizer.model_max_length is None or tokenizer.model_max_length > max_generate_length:
        tokenizer.model_max_length = max_generate_length

    generation_kwargs = {
        "top_p": 0.95,
        "temperature": 0.8,
        "max_length": max_generate_length,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "early_stopping": True,
        "no_repeat_ngram_size": 4,
    }

    sess_text = ""
    while True:
        raw_text = input("prompt(\"exit\" to end, \"clear\" to clear session) >>> ")
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
        input_text = prompt_input.format_map({'instruction': sess_text.split(tok_ins, 1)[1]})
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=max_input_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model.generate(**inputs, **generation_kwargs)
        output_str = tokenizer.decode(output[0], skip_special_tokens=False, spaces_between_special_tokens=False)
        answer = output_str.rsplit(tok_res, 1)[1].strip()
        if answer.endswith(tokenizer.eos_token):
            answer = answer.rsplit(tokenizer.eos_token, 1)[0].strip()
        sess_text += tok_res + answer

        print("=" * 100)
        print(answer)
        print("=" * 100)


class W8A16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, quant_w: torch.Tensor, scale_w: torch.Tensor, weight_bit_width):
        ctx.inp_shape = inp.size()
        ctx.weight_bit_width = weight_bit_width
        out_features = quant_w.size(0)
        inp = inp.contiguous().view(-1, inp.size(-1))
        weight = extract_weight_to_half(quant_w, scale_w, weight_bit_width)
        ctx.weight_shape = weight.size()
        output = inp.mm(weight.t())
        ctx.save_for_backward(inp, quant_w, scale_w)
        return output.view(*(ctx.inp_shape[:-1] + (out_features,)))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inp, quant_w, scale_w = ctx.saved_tensors
        weight = extract_weight_to_half(quant_w, scale_w, ctx.weight_bit_width)
        grad_output = grad_output.contiguous().view(-1, weight.size(0))
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(inp)
        return grad_input.view(ctx.inp_shape), grad_weight.view(ctx.weight_shape), None, None


def compress_int4_weight(weight: torch.Tensor):  # (n, m)
    with torch.cuda.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        assert m % 2 == 0
        m = m // 2
        out = torch.empty(n, m, dtype=torch.int8, device="cuda")
        stream = torch.cuda.current_stream()

        gridDim = (n, 1, 1)
        blockDim = (min(round_up(m, 32), 1024), 1, 1)

        kernels.int4WeightCompression(
            gridDim,
            blockDim,
            0,
            stream,
            [ctypes.c_void_p(weight.data_ptr()), ctypes.c_void_p(out.data_ptr()), ctypes.c_int32(n), ctypes.c_int32(m)],
        )
        return out


def extract_weight_to_half(weight: torch.Tensor, scale_list: torch.Tensor, source_bit_width: int):
    if source_bit_width == 8:
        func = kernels.int8WeightExtractionHalf
    elif source_bit_width == 4:
        func = kernels.int4WeightExtractionHalf
    else:
        assert False, "Unsupported bit-width"

    with torch.cuda.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        out = torch.empty(n, m * (8 // source_bit_width), dtype=torch.half, device="cuda")
        stream = torch.cuda.current_stream()

        gridDim = (n, 1, 1)
        blockDim = (min(round_up(m, 32), 1024), 1, 1)

        func(
            gridDim,
            blockDim,
            0,
            stream,
            [
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(scale_list.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int32(n),
                ctypes.c_int32(m),
            ],
        )
        return out


class QuantizedLinear(Linear):
    def __init__(self, weight_bit_width: int, weight_tensor=None, bias_tensor=None, empty_init=False, *args, **kwargs):
        super(QuantizedLinear, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width

        shape = self.weight.shape
        del self.weight

        if weight_tensor is None or empty_init:
            self.weight = torch.empty(
                shape[0], shape[1] * weight_bit_width // 8, dtype=torch.int8, device=kwargs["device"]
            )
            self.weight_scale = torch.empty(shape[0], dtype=kwargs["dtype"], device=kwargs["device"])
        else:
            self.weight_scale = (weight_tensor.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)).half()
            self.weight = torch.round(weight_tensor / self.weight_scale[:, None]).to(torch.int8)
            if weight_bit_width == 4:
                self.weight = compress_int4_weight(self.weight)

        self.weight = Parameter(self.weight.to(kwargs["device"]), requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale.to(kwargs["device"]), requires_grad=False)
        if bias_tensor is not None:
            self.bias = Parameter(bias_tensor.to(kwargs["device"]), requires_grad=False)
        else:
            self.bias = None

    def forward(self, input):
        output = W8A16Linear.apply(input, self.weight, self.weight_scale, self.weight_bit_width)
        if self.bias is not None:
            output = output + self.bias
        return output


def quantize(model, weight_bit_width, empty_init=False, **kwargs):
    """Replace fp16 linear with quantized linear"""

    # print(model.model.layers)
    for layer in model.model.layers:
        layer.self_attn.q_proj = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight_tensor=layer.self_attn.q_proj.weight.to(torch.cuda.current_device()),
            bias_tensor=layer.self_attn.q_proj.bias,
            in_features=layer.self_attn.q_proj.in_features,
            out_features=layer.self_attn.q_proj.out_features,
            bias=True,
            dtype=torch.half,
            device=layer.self_attn.q_proj.weight.device,
            empty_init=empty_init
        )
        layer.self_attn.k_proj = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight_tensor=layer.self_attn.k_proj.weight.to(torch.cuda.current_device()),
            bias_tensor=layer.self_attn.k_proj.bias,
            in_features=layer.self_attn.k_proj.in_features,
            out_features=layer.self_attn.k_proj.out_features,
            bias=True,
            dtype=torch.half,
            device=layer.self_attn.k_proj.weight.device,
            empty_init=empty_init
        )
        layer.self_attn.v_proj = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight_tensor=layer.self_attn.v_proj.weight.to(torch.cuda.current_device()),
            bias_tensor=layer.self_attn.v_proj.bias,
            in_features=layer.self_attn.v_proj.in_features,
            out_features=layer.self_attn.v_proj.out_features,
            bias=True,
            dtype=torch.half,
            device=layer.self_attn.v_proj.weight.device,
            empty_init=empty_init
        )
        layer.self_attn.o_proj = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight_tensor=layer.self_attn.o_proj.weight.to(torch.cuda.current_device()),
            bias_tensor=layer.self_attn.o_proj.bias,
            in_features=layer.self_attn.o_proj.in_features,
            out_features=layer.self_attn.o_proj.out_features,
            bias=True,
            dtype=torch.half,
            device=layer.self_attn.o_proj.weight.device,
            empty_init=empty_init
        )

        layer.mlp.gate_proj = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight_tensor=layer.mlp.gate_proj.weight.to(torch.cuda.current_device()),
            bias_tensor=layer.mlp.gate_proj.bias,
            in_features=layer.mlp.gate_proj.in_features,
            out_features=layer.mlp.gate_proj.out_features,
            bias=True,
            dtype=torch.half,
            device=layer.mlp.gate_proj.weight.device,
            empty_init=empty_init
        )
        layer.mlp.down_proj = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight_tensor=layer.mlp.down_proj.weight.to(torch.cuda.current_device()),
            bias_tensor=layer.mlp.down_proj.bias,
            in_features=layer.mlp.down_proj.in_features,
            out_features=layer.mlp.down_proj.out_features,
            bias=True,
            dtype=torch.half,
            device=layer.mlp.down_proj.weight.device,
            empty_init=empty_init
        )
        layer.mlp.up_proj = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight_tensor=layer.mlp.up_proj.weight.to(torch.cuda.current_device()),
            bias_tensor=layer.mlp.up_proj.bias,
            in_features=layer.mlp.up_proj.in_features,
            out_features=layer.mlp.up_proj.out_features,
            bias=True,
            dtype=torch.half,
            device=layer.mlp.up_proj.weight.device,
            empty_init=empty_init
        )

    return model


if __name__ == "__main__":
    fire.Fire(main)
