# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any, Dict, List, Optional, Union

import torch
from omegaconf import DictConfig
from torch import nn

from torchtune import config, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import ChatFormat, InstructTemplate, Message
import autonvtx

logger = utils.get_logger("DEBUG")


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        torch.cuda.nvtx.range_push("InferenceRecipe::__init__")
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype, device=self._device)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = utils.get_quantizer_mode(self._quantizer)

        utils.set_seed(seed=cfg.seed)
        torch.cuda.nvtx.range_pop()

    def setup(self, cfg: DictConfig) -> None:
        torch.cuda.nvtx.range_push("InferenceRecipe::setup")
        
        torch.cuda.nvtx.range_push("checkpointer_setup")
        checkpointer = config.instantiate(cfg.checkpointer)
        print("Checkpointer instantiated")
        if self._quantization_mode is None:
            print("Loading checkpoint")
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            print("Loading quantized checkpoint")
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)
        # Print current memory usage
        print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push("model_setup")
        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
            enable_kv_cache=cfg.enable_kv_cache,
        )
        # Clear cuda cache after model setup
        torch.cuda.empty_cache()
        
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push("tokenizer_setup")
        self._tokenizer = config.instantiate(cfg.tokenizer)
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_pop()
        
    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
        enable_kv_cache: bool = True,
    ) -> nn.Module:
        torch.cuda.nvtx.range_push("InferenceRecipe::_setup_model")
        torch.cuda.nvtx.range_push("model_instantiation")
        # with utils.set_default_dtype(self._dtype), self._device:
        #     model = config.instantiate(model_cfg)
        model = config.instantiate(model_cfg)
        torch.cuda.nvtx.range_pop()
        
        # Print current memory usage
        print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
        
        if self._quantization_mode is not None:
            print("Quantizing model")
            torch.cuda.nvtx.range_push("quantization")
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)
            torch.cuda.nvtx.range_pop()
        else:
            # Move the model to the device
            model = model.to(device=self._device, dtype=self._dtype)    
        
        # Print current memory usage
        print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

        
        torch.cuda.nvtx.range_push("model_load_state_dict")
        model.load_state_dict(model_state_dict)
        torch.cuda.nvtx.range_pop()

        # Validate model was loaded in with the expected dtype.
        torch.cuda.nvtx.range_push("validate_dtype")
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        torch.cuda.nvtx.range_pop()
        logger.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        if enable_kv_cache:
            torch.cuda.nvtx.range_push("setup_caches")
            with self._device:
                model.setup_caches(batch_size=1, dtype=self._dtype)
            torch.cuda.nvtx.range_pop()
            
        # Wrap the model with autonvtx for NVTX profiling
        model = autonvtx(model)

        torch.cuda.nvtx.range_pop()

        return model

    def convert_prompt_to_tokens(
        self,
        prompt: Union[DictConfig, str],
        chat_format: Optional[ChatFormat],
        instruct_template: Optional[InstructTemplate],
    ) -> List[Message]:
        """
        Either:
        (1) a raw string is passed as the prompt, in which case we call tokenizer.encode directly, or
        (2) a DictConfig is passed as the prompt. In this case there are three possibilities:
            (a) an InstructTemplate is provided. Since instruct templates output a string, we will
                call tokenizer.encode on the output of the instruct template.
            (b) a ChatFormat is provided. Since chat formats output a list of messages, we will
                call tokenizer.tokenize_messages on the output of the chat format.
            (c) neither an InstructTemplate nor a ChatFormat is provided. In this case we will
                convert the DictConfig to a list of messages and call tokenizer.tokenize_messages directly.
        """

        # Should only be chat-style prompt or instruct-style prompt
        if chat_format and instruct_template:
            raise ValueError(
                "Cannot pass both chat format and instruct template for generation"
            )

        # If instruct template is provided, assert that the prompt is a DictConfig
        # and apply it
        if instruct_template:
            if not isinstance(prompt, DictConfig):
                raise ValueError("Cannot apply instruct template to raw string")
            instruct_template = _get_component_from_path(instruct_template)
            prompt = instruct_template.format(prompt)

        # To hit this block, either the raw prompt is a string or an
        # instruct template has been provided to convert it to a string
        if isinstance(prompt, str):
            return self._tokenizer.encode(prompt, add_bos=True, add_eos=False)

        # dict.items() will respect order for Python >= 3.7
        else:
            messages = [Message(role=k, content=v) for k, v in prompt.items()]
            messages += [Message(role="assistant", content="")]
            if chat_format:
                chat_format = _get_component_from_path(chat_format)
                messages = chat_format.format(messages)
            return self._tokenizer.tokenize_messages(messages)[0]
        
    @torch.no_grad()
    def generate(self, cfg: DictConfig):
        torch.cuda.nvtx.range_push("convert_prompt_to_tokens")
        tokens = self.convert_prompt_to_tokens(
            cfg.prompt, cfg.get("chat_format", None), cfg.get("instruct_template", None)
        )
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push("prompt_tensor")
        prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)
        torch.cuda.nvtx.range_pop()
        
        
        custom_generate_next_token = None

        t0 = time.perf_counter()
        logger.info("Starting generation ...")
        torch.cuda.nvtx.range_push("generate")
        generated_tokens = utils.generate(
            model=self._model,
            prompt=prompt,
            max_generated_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            stop_tokens=self._tokenizer.stop_tokens,
            pad_id=self._tokenizer.pad_id,
            custom_generate_next_token=custom_generate_next_token,
        )
        torch.cuda.nvtx.range_pop()
        t = time.perf_counter() - t0

        logger.info(self._tokenizer.decode(generated_tokens[0]))

        torch.cuda.nvtx.range_push("calculate_model_size")
        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(
                    self._model.parameters(), self._model.buffers()
                )
            ]
        )
        torch.cuda.nvtx.range_pop()

        tokens_generated = len(generated_tokens[0]) - prompt.size(0)
        tokens_sec = tokens_generated / t
        logger.info(
            f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        logger.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    # recipe.generate(cfg=cfg)
    torch.cuda.profiler.start()
    with torch.autograd.profiler.emit_nvtx():
        recipe.generate(cfg=cfg)
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    sys.exit(main())
