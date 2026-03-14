import torch
import torch_npu
import time
import os
import argparse
from stablediffusion import StableDiffusionPipeline
from torch_npu.contrib import transfer_to_npu
torch.npu.config.allow_internal_format = False
torch_npu.npu.set_compile_mode(jit_compile=False)

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
        "--prompt_file",
        type=str,
        default="./prompts/prompts.txt",
        help="The prompts file to guide images generation.",
    )
    parser.add_argument(
        "--prompt_file_type",
        choices=["plain", "parti"],
        default="plain",
        help="Type of prompt file.",
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
        "-bs",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Path to save result audio files.",
    )
    parser.add_argument(
        "--enable_dp",
        action="store_true",
        help="Enable dp parallel.",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable dp parallel.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="./lora",
        help="The path of lora model weights.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    prompt_loader = PromptLoader(args.prompt_file,
                                 args.prompt_file_type,
                                 args.batch_size)

    if args.enable_dp:
        from stablediffusion.parallel.parallel_config import ParallelCfg
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank_idx = int(os.environ["LOCAL_RANK"])
        parallel_cfg = ParallelCfg(enable_dp=args.enable_dp, device_id=local_rank_idx, local_rank=local_rank_idx, world_size=world_size)
    else:
        parallel_cfg = None
        torch_npu.npu.set_device(args.device)

    torch.manual_seed(1)
    npu_stream = torch_npu.npu.Stream()

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    if args.use_lora:
        pipe.load_lora_weights(args.lora_path)
        pipe.fuse_lora()
    pipe.to("npu")

    use_time = 0
    infer_num = 0
    image_info = []
    current_prompt = None
    for i, input_info in enumerate(prompt_loader):
        prompts = input_info['prompts']
        catagories = input_info['catagories']
        save_names = input_info['save_names']
        n_prompts = input_info['n_prompts']

        print(f"[{infer_num + n_prompts}/{len(prompt_loader)}]: {prompts}")
        infer_num += args.batch_size
        with torch.no_grad():
            npu_stream.synchronize()
            begin = time.time()
            images = pipe(
                prompt=prompts,
                num_inference_steps=args.steps,
                parallel_cfg=parallel_cfg,
            )
            npu_stream.synchronize()
            end = time.time()
            if i > 4:
                use_time += end - begin
        for j in range(n_prompts):
            image_save_path = os.path.join(save_dir, f"{save_names[j]}.png")
            image = images[0][j]
            image.save(image_save_path)

            if current_prompt != prompts[j]:
                current_prompt = prompts[j]
                image_info.append({'images': [], 'prompt': current_prompt, 'category': catagories[j]})

            image_info[-1]['images'].append(image_save_path)
    infer_num = infer_num - 5 # do not count the time spent inferring the first 5 images
    print(f"[info] infer number: {infer_num}; use time: {use_time:.3f}s\n"
          f"average time: {use_time / infer_num:.3f}s\n")


if __name__ == "__main__":
    main()