#!/bin/bash

set -euo pipefail

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="./logs/train_all_${timestamp}"
pid_file="${log_dir}/pids.tsv"

mkdir -p "$log_dir"
printf "pid\tjob\tlog\n" > "$pid_file"

launch_job() {
	local job_name="$1"
	shift

	local log_file="${log_dir}/${job_name}.log"

	nohup "$@" >"$log_file" 2>&1 < /dev/null &
	local pid=$!

	printf "%s\t%s\t%s\n" "$pid" "$job_name" "$log_file" >> "$pid_file"
	echo "started ${job_name}: pid=${pid}, log=${log_file}"
}

launch_job cyberpunk_all bash training_scripts/run_lora_pti_sdxl_single_cyberpunk_all.sh
launch_job wushan env CUDA_VISIBLE_DEVICES=1 bash training_scripts/run_lora_pti_sdxl_single_wushan.sh
launch_job cyberpunk_portrait env CUDA_VISIBLE_DEVICES=2 bash training_scripts/run_lora_pti_sdxl_single_cyberpunk_portrait.sh
launch_job jojo env CUDA_VISIBLE_DEVICES=3 bash training_scripts/run_lora_pti_sdxl_single_jojo.sh
launch_job mygo env CUDA_VISIBLE_DEVICES=4 bash training_scripts/run_lora_pti_sdxl_single.sh

echo "all jobs launched in background"
echo "pid list: ${pid_file}"
echo "log dir: ${log_dir}"