import os
import csv
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

import torch
import torch_npu

from stablediffusion import StableDiffusionPipeline


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="./prompts/prompts.txt",
        help="The prompts file to guide images generation.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="The prompt or prompts to guide what to not include in image generation.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="The number of denoising steps.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./stable-diffusion-v1.5",
        help="The path of stable-diffusion.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="NPU device id.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Path to save result audio files.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=4,
        help="Number of images generated for each prompt.",
    )
    return parser.parse_args()

def load_parti(file_path):
    prompts = []
    with os.fdopen(os.open(file_path, os.O_RDONLY), "r", encoding="utf8") as f:
        next(f)
        tsv_file = csv.reader(f, delimiter="\t")
        for image_id, line in enumerate(tsv_file):
            prompt = line[0]
            prompts.append((prompt, image_id))
    return prompts
def main():
    args = parse_arguments()
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_time = time.time()
    prompts_loader = load_parti(args.prompt_file)
    infer_nums = len(prompts_loader * args.num_images_per_prompt)
    save_executor = ThreadPoolExecutor(max_workers=2)
    futures = []
    torch_npu.npu.set_device(args.device)
    torch.manual_seed(1)
    npu_stream = torch_npu.npu.Stream()

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe.to("npu").to(torch.float16)

    for i, prompt_info in enumerate(prompts_loader):
        prompt, image_id = prompt_info
        print(f"[{i}/{len(prompts_loader)}]: {prompt}")
        with torch.no_grad():
            images = pipe(
                prompt=prompt,
                num_images_per_prompt=args.num_images_per_prompt,
                um_inference_steps=args.steps,
                ).images
        image_save_paths = [
            os.path.join(args.save_dir, "{:05d}".format(image_id * args.num_images_per_prompt + i) + ".png") \
            for i in range(args.num_images_per_prompt)
        ]
        for j in range(args.num_images_per_prompt):
            futures.append(save_executor.submit(images[j].save, image_save_paths[j]))

    total_time = time.time() - start_time
    print(f"[info] infer number: {infer_nums}; use time: {total_time:.3f}s\n")
    print(f"Infer average time: {total_time / infer_nums:.3f}s\n")


if __name__ == "__main__":
    main()