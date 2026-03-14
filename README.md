---
license: apache-2.0
pipeline_tag: text-to-image
frameworks:
  - PyTorch
library_name: openmind
hardwares:
  - NPU
language:
  - en
---
## 一、准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10 / 3.11 | - |
  | torch | 2.1.0 | - |

### 1.1 获取CANN&MindIE安装包&环境准备
- 设备支持
Atlas 800I A2推理设备：支持的卡数为1或2
Atlas 300I Duo推理卡：支持的卡数为1，可双芯并行
- [Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- [Atlas 300I Duo](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=2&model=17)
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/softwareinst/instg/instg_0001.html)

### 1.2 CANN安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构，{soc}表示昇腾AI处理器的版本。
chmod +x ./Ascend-cann-toolkit_{version}_linux-{arch}.run
chmod +x ./Ascend-cann-kernels-{soc}_{version}_linux.run
# 校验软件包安装文件的一致性和完整性
./Ascend-cann-toolkit_{version}_linux-{arch}.run --check
./Ascend-cann-kernels-{soc}_{version}_linux.run --check
# 安装
./Ascend-cann-toolkit_{version}_linux-{arch}.run --install
./Ascend-cann-kernels-{soc}_{version}_linux.run --install

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```


### 1.3 MindIE安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x ./Ascend-mindie_${version}_linux-${arch}.run
./Ascend-mindie_${version}_linux-${arch}.run --check

# 方式一：默认路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install
# 设置环境变量
cd /usr/local/Ascend/mindie && source set_env.sh

# 方式二：指定路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install-path=${AieInstallPath}
# 设置环境变量
cd ${AieInstallPath}/mindie && source set_env.sh
```

### 1.4 Torch_npu安装
安装pytorch框架 版本2.1.0
[安装包下载](https://download.pytorch.org/whl/cpu/torch/)

使用pip安装
```shell
# {version}表示软件版本号，{arch}表示CPU架构。
pip install torch-${version}-cp310-cp310-linux_${arch}.whl
```
下载 pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
```shell
tar -xzvf pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
# 解压后，会有whl包
pip install torch_npu-{pytorchversion}.xxxx.{arch}.whl
```
## 二、下载本仓库

### 2.1 下载到本地
```shell
git clone https://modelers.cn/MindIE/stable_diffusion_v1.5.git
```

### 2.2 环境依赖安装
```bash
pip3 install -r requirements.txt
```

## 三、Stable-Diffusion-v1.5 使用

### 3.1 权重及配置文件说明
下载权重和配置文件
```shell
# stable-diffusion-v1.5:
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
```

### 3.2 修改配置文件
将model_index.json中所有的`diffusers`字段修改为`stablediffusion`

### 3.3 执行算子编译脚本
300I Duo机器一定要执行该步骤，800I A2机器不需要执行
```shell
cd pta_plugin
bash build.sh
```

### 3.4 性能测试
设置权重路径
```shell
model_base='./stable-diffusion-v1-5'
```
执行命令：
```shell
# 800I A2，单卡推理
export TOKEN_DOWNSAMPLE=1
export ENABLE_CACHE=1
python3 inference_stablediffusion.py \
        --model ${model_base} \
        --prompt_file ./prompts/prompts.txt \
        --steps 50 \
        --batch_size 1 \
        --save_dir ./results \
        --device 0
# 300I Duo，使能单卡双芯
export FATIK_FILE_PATH=./pta_plugin/build/libPTAExtensionOPS.so
export ENABLE_CACHE=1
export TOKEN_DOWNSAMPLE=1
torchrun --nproc_per_node 2 inference_stablediffusion.py \
        --model ${model_base} \
        --prompt_file ./prompts/prompts.txt \
        --steps 50 \
        --save_dir ./results_dp \
        --enable_dp
```
参数说明：
- --TOKEN_DOWNSAMPLE：设置为1使能序列压缩优化；设置为0不使能
- --ENABLE_CACHE：设置为1使能cache优化；设置为0不使能cache优化
- --FATIK_FILE_PATH：300I Duo机器需要设置；800I A2机器不用设置
- --model：模型权重路径。
- --prompt_file：提示词文件。
- --steps: 图片生成迭代次数。
- --batch_size：模型batch size。
- --save_dir：生成图片的存放目录。
- --device：推理设备ID。
- --enable_dp：使能dp并行。

### 3.5 lora热切换
下载lora模型的权重，设置权重路径
```shell
model_base='./stable-diffusion-v1-5'
lora_base='./pytorch_lora_weights.safetensors'
```
执行命令
```shell
# 800I A2，单卡推理
export TOKEN_DOWNSAMPLE=1
export ENABLE_CACHE=1
python3 inference_stablediffusion.py \
        --model ${model_base} \
        --prompt_file ./prompts/prompts.txt \
        --steps 50 \
        --batch_size 1 \
        --save_dir ./results \
        --device 0 \
        --use_lora \
        --lora_path ${lora_base}
```
参数说明：
- --use_lora：开启lora热切换
- --lora_path：Lora模型权重路径

### 3.5 模型推理性能

性能参考下列数据。

| 硬件形态 | 迭代次数 | 平均耗时|
| :------: |:----:|:----:|:----:|
| Atlas 800I A2(8*32G) |  50  |  2.821s |

### 3.6 模型精度验证
   本章节将使用Parti数据集对Stable-Diffusion-v1.5进行精度验证。
   由于生成的图片存在随机性，所以精度验证将使用CLIP-score来评估图片和输入文本的相关性，分数的取值范围为[-1, 1]，越高越好。

   注意，由于要生成的图片数量较多，进行完整的精度验证需要耗费很长的时间。

   1. 下载Parti数据集

      ```bash
      wget https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv --no-check-certificate
      ```

   2. 下载Clip模型权重

      ```bash
      # 安装git-lfs
      apt install git-lfs
      git lfs install
      git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K

      # 或者访问https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/open_clip_pytorch_model.bin，将权重下载并放到这个目录下
      ```

   3. 使用推理脚本读取Parti数据集，生成图片
      ```bash
      export ENABLE_CACHE=1
      python inference_stablediffusion_parti.py \
             --model ${model_base} \
             --prompt_file ./PartiPrompts.tsv \
             --num_images_per_prompt 4 \
             --steps 50 \
             --batch_size 1 \
             --save_dir ./results \
             --device 0
      ```
      增加的参数说明：
      - --num_images_per_prompt: 每个prompt生成的图片数量。

      执行完成后会在`./results`目录下生成推理图片。

   4. 计算CLIP-score
      设置CLIP权重路径
      ```bash
      clip_model_base='./open_clip_pytorch_model.bin'
      ```
      ```bash
      ASCEND_RT_VISIBLE_DEVICES=0 python clip_score_parti.py \
             --device="npu" \
             --model_name="ViT-H-14" \
             --model_weights_path=${clip_model_base} \
             --prompt_file="./PartiPrompts.tsv" \
             --image_prefix="./results"
      ```

      参数说明：
      - --device: 推理设备。
      - --model_name: Clip模型名称。
      - --model_weights_path: Clip模型权重文件路径。
      - --prompt_file: 提示词文件。
      - --image_prefix: 生成图片的存放路径。

      执行完成后会在屏幕打印出精度计算结果。 

## 优化指南
本模型使用的优化手段如下：
- 等价优化：FA、DP并行
- 有损优化：cache

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。