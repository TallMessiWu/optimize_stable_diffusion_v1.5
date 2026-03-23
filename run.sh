echo performance |tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0


export FATIK_FILE_PATH=./pta_plugin/build/libPTAExtensionOPS.so
export ENABLE_CACHE=1
export TOKEN_DOWNSAMPLE=1
export TASK_QUEUE_ENABLE=2
export CACHE_LEVEL=4

export GATHER_PROFILING=1

export model_base="/home/stable_diffusion"
export ASCEND_RT_VISIBLE_DEVICES=4,5

numactl -C 48-71 torchrun --nproc_per_node 2 \
        --master_addr localhost \
        --master_port 6000 \
        inference_stablediffusion.py \
        --model ${model_base}/stable-diffusion-v1-5 \
        --prompt_file ./prompts/prompts.txt \
        --steps 50 \
        --batch_size 1 \
        --save_dir ./results_compile \
        --enable_dp \
        --use_compile