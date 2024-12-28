## 一、准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10.2 | - |
  | torch | 2.1.0 | - |

### 1.1 获取CANN&MindIE安装包&环境准备
- [800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- [Duo卡](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=2&model=17)
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

### 3.3 单卡功能测试
设置权重路径
```shell
model_base='./stable-diffusion-v1-5'
```
执行命令：
```shell
python3 inference_stablediffusion.py \
        --model ${model_base} \
        --prompt_file ./prompts/prompts.txt \
        --steps 50 \
        --save_dir ./results \
        --device 0
```
参数说明：
- --model：模型权重路径。
- --prompt_file：提示词文件。
- --steps: 图片生成迭代次数。
- --save_dir：生成图片的存放目录。
- --device：推理设备ID。


### 3.4 模型推理性能

性能参考下列数据。

| 硬件形态 | 迭代次数 | 平均耗时|
| :------: |:----:|:----:|
| Atlas 800I A2 (32G) |  50  |  2.658s  |