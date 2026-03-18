# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import time
import logging
import csv
import json

from typing import Callable, List, Optional, Union
import numpy as np

import torch
import torch_npu

from stablediffusion import StableDiffusionPipeline
from stablediffusion.layers.attention_processor import soc
torch.npu.config.allow_internal_format = False
torch_npu.npu.set_compile_mode(jit_compile=False)

if soc == "DUO":
    torch.npu.config.allow_internal_format = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class PromptLoader:
    def __init__(
            self,
            prompt_file: str,
            prompt_file_type: str,
            batch_size: int,
            num_images_per_prompt: int = 1,
            max_num_prompts: int = 0
    ):
        self.prompts = []
        self.catagories = ['Not_specified']
        self.batch_size = batch_size
        self.num_images_per_prompt = num_images_per_prompt

        if prompt_file_type == 'plain':
            self.load_prompts_plain(prompt_file, max_num_prompts)
        elif prompt_file_type == 'parti':
            self.load_prompts_parti(prompt_file, max_num_prompts)
        elif prompt_file_type == 'hpsv2':
            self.load_prompts_hpsv2(max_num_prompts)
        else:
            print("This operation is not supported!")

        self.current_id = 0
        self.inner_id = 0

    def __len__(self):
        return len(self.prompts) * self.num_images_per_prompt

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_id == len(self.prompts):
            raise StopIteration

        ret = {
            'prompts': [],
            'catagories': [],
            'save_names': [],
            'n_prompts': self.batch_size,
        }
        for _ in range(self.batch_size):
            if self.current_id == len(self.prompts):
                ret['prompts'].append('')
                ret['save_names'].append('')
                ret['catagories'].append('')
                ret['n_prompts'] -= 1

            else:
                prompt, catagory_id = self.prompts[self.current_id]
                ret['prompts'].append(prompt)
                ret['catagories'].append(self.catagories[catagory_id])
                ret['save_names'].append(f'{self.current_id}_{self.inner_id}')

                self.inner_id += 1
                if self.inner_id == self.num_images_per_prompt:
                    self.inner_id = 0
                    self.current_id += 1

        return ret

    def load_prompts_plain(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            for i, line in enumerate(f):
                if max_num_prompts and i == max_num_prompts:
                    break

                prompt = line.strip()
                self.prompts.append((prompt, 0))

    def load_prompts_parti(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            # Skip the first line
            next(f)
            tsv_file = csv.reader(f, delimiter="\t")
            for i, line in enumerate(tsv_file):
                if max_num_prompts and i == max_num_prompts:
                    break

                prompt = line[0]
                catagory = line[1]
                if catagory not in self.catagories:
                    self.catagories.append(catagory)

                catagory_id = self.catagories.index(catagory)
                self.prompts.append((prompt, catagory_id))

    def load_prompts_hpsv2(self, max_num_prompts: int):
        with open('hpsv2_benchmark_prompts.json', 'r') as file:
            all_prompts = json.load(file)
        count = 0
        for style, prompts in all_prompts.items():
            for prompt in prompts:
                count += 1
                if max_num_prompts and count >= max_num_prompts:
                    break

                if style not in self.catagories:
                    self.catagories.append(style)

                catagory_id = self.catagories.index(style)
                self.prompts.append((prompt, catagory_id))



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="./stable-diffusion-2-1-base",
        help="The path of stable-diffusion.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="The number of denoising steps.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="NPU device id.",
    )
    parser.add_argument(
        "--dtype",
        type=torch.dtype,
        default=torch.float16
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="./prompts/prompts.txt",
        help="A text file of prompts for generating images.",
    )
    parser.add_argument(
        "--prompt_file_type",
        choices=["plain", "parti", "hpsv2"],
        default="plain",
        help="Type of prompt file.",
    )
    return parser.parse_args()


def test_precision():
    args = parse_arguments()

    torch.npu.set_device(args.device)

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe.to(args.dtype).to("npu")

    prompt_loader = PromptLoader(
        args.prompt_file,
        args.prompt_file_type,
        batch_size=1,
        num_images_per_prompt=args.num_images_per_prompt
    )
    image_info = []
    current_prompt = None
    infer_num = 0

    current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    img_dir = f"{current_time}/images"
    os.makedirs(img_dir)

    for i, input_info in enumerate(prompt_loader):
        print(f"==============={i}=================", input_info)
        prompts = input_info['prompts']
        catagories = input_info['catagories']
        save_names = input_info['save_names']
        n_prompts = input_info['n_prompts']

        logger.info(f"[{infer_num + n_prompts}/{len(prompt_loader)}]: {prompts}")

        infer_num += 1

        images = pipe(
            prompt=prompts,
            negative_prompt=[""],
            num_inference_steps=args.steps,
        )

        for j in range(n_prompts):
            image_save_path = os.path.join(img_dir, f"{save_names[j]}.png")
            image = images[0][j]
            image.save(image_save_path)

            if current_prompt != prompts[j]:
                current_prompt = prompts[j]
                image_info.append({'images': [], 'prompt': current_prompt, 'category': catagories[j]})

            image_info[-1]['images'].append(image_save_path)
    
    img_json = f"{current_time}/image_info.json"
    
    with os.fdopen(os.open(img_json, os.O_RDWR | os.O_CREAT, 0o640), "w") as f:
        json.dump(image_info, f)


if __name__ == "__main__":
    test_precision()