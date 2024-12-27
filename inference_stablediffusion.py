import torch
import torch_npu
import time
import os
import argparse
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
        default=100,
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
    return parser.parse_args()


def main():
    args = parse_arguments()
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch_npu.npu.set_device(args.device)
    torch.manual_seed(1)
    npu_stream = torch_npu.npu.Stream()

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe.to("npu")

    total_time = 0
    prompts_num = 0
    average_time = 0
    skip = 2
    with os.fdopen(os.open(args.prompt_file, os.O_RDONLY), "r") as f:
        for i, prompt in enumerate(f):
            with torch.no_grad():
                npu_stream.synchronize()
                begin = time.time()
                image = pipe(prompt).images[0]
                npu_stream.synchronize()
                end = time.time()
                if i > skip - 1:
                    total_time += end - begin
            prompts_num = i+1
            image_save_path = os.path.join(save_dir, f"images_{i}.png")
            image.save(image_save_path)
    if prompts_num > skip:
        average_time = total_time / (prompts_num-skip)
    else:
        raise ValueError("Infer average time skip first two prompts, ensure that prompts.txt \
                         contains more than three prompts")
    print(f"Infer average time: {average_time:.3f}s\n")


if __name__ == "__main__":
    main()