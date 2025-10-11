# Copyright 2023 Huawei Technologies Co., Ltd
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
import csv
import json
import time
import argparse

import open_clip
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    print("torch_npu not found.")

def load_parti(file_path):
    prompts = []
    with os.fdopen(os.open(file_path, os.O_RDONLY), "r", encoding="utf8") as f:
        next(f)
        tsv_file = csv.reader(f, delimiter="\t")
        for image_id, line in enumerate(tsv_file):
            prompt = line[0]
            prompts.append((prompt, image_id))
    return prompts

def clip_score(model_clip, tokenizer, preprocess, prompt, image_file, device):
    img = preprocess(Image.open(image_file)).unsqueeze(0).to(device)
    text = tokenizer(prompt).to(device)

    with torch.no_grad():
        text_ft = model_clip.encode_text(text)
        img_ft = model_clip.encode_image(img)
        score = F.cosine_similarity(img_ft, text_ft)
    
    return score.item()


def main():
    args = parse_arguments()
    repeat = args.num_images_per_prompt

    if args.device is None:
        try:
            device = torch.device("npu")
        except:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    t_b = time.time()
    print(f"Load clip model...") 
    model_clip, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.model_weights_path, device=device)
    model_clip.eval()
    print(f">done. elapsed time: {(time.time() - t_b):.3f} s")
    
    tokenizer = open_clip.get_tokenizer(args.model_name)

    all_scores = []
    prompts_loader = load_parti(args.prompt_file)
    for i, prompt_info in enumerate(prompts_loader):
        scores = []
        prompt, image_id = prompt_info
        print(f"[{i}/{len(prompts_loader)}]: {prompt}")
        for j in range(repeat):
            image_file = os.path.join(args.image_prefix, "{:05d}".format(i * repeat + j) + ".png")
            image_score = clip_score(model_clip, 
                                    tokenizer, 
                                    preprocess, 
                                    prompt, 
                                    image_file, 
                                    device)
            scores.append(image_score)
        best_score = np.max(scores)
        print(f"image scores: {scores}")
        print(f"best score: {best_score}")
        all_scores.append(best_score)
        
    average_score = np.average(all_scores)
    print(f"====================================")
    print(f"average score: {average_score:.3f}")
    

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="npu",
        choices=["cpu", "cuda", "npu"],
        help="device for torch.",
    )
    parser.add_argument(
        "--image_prefix",
        type=str,
        default="./results",
        help="path of generated images.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-H-14",
        help="open clip model name",
    )
    parser.add_argument(
        "--model_weights_path",
        type=str,
        default="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
        help="open clip model weights",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=4,
        help="Number of images generated for each prompt.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="./prompts/prompts.txt",
        help="The prompts file to guide images generation.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()