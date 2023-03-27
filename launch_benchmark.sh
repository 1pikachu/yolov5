#!/bin/bash
set -xe

function main {
    source oob-common/common.sh
    # set common info
    init_params $@
    fetch_device_info
    set_environment

    # requirements

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    if [[ "${mode_name}" == "train" ]];then
    	rm -rf datasets && mkdir -p datasets
    	cp -r ${DATASET_DIR} datasets/.
	exec_cmd=" train.py --data data/coco128.yaml --weights '' \
		--cfg models/yolov5x.yaml --device ${device} "
    else
	exec_cmd=" detect.py --weights ${CKPT_DIR} --source ${DATASET_INF_DIR} \
		--jit --num_iter ${num_iter} --num_warmup ${num_warmup} \
		--device_str ${device} "
    fi
    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        #
        for batch_size in ${batch_size_list[@]}
        do
            # clean workspace
            logs_path_clean

            # generate launch script for multiple instance
            generate_core
            # launch
            echo -e "\n\n\n\n Running..."
            source ${excute_cmd_file}
            echo -e "Finished.\n\n\n\n"
            # collect launch result
            collect_perf_logs
        done
    done
}

# run
function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        # instances
        if [ "${device}" == "cpu" ];then
            OOB_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            OOB_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        elif [ "${device}" == "cuda" ];then
            OOB_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
	    if [[ "${mode_name}" == "realtime" ]];then
	        addtion_options+=" --nv_fuser "
	    fi
	fi
        printf " ${OOB_EXEC_HEADER} \
	    python ${exec_cmd} \
	        --batch-size ${batch_size} --precision ${precision} \
	        --channels_last ${channels_last} \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        if [ "${numa_nodes_use}" == "0" ];then
            break
        fi
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# download common files
rm -rf oob-common && git clone https://github.com/intel-sandbox/oob-common.git -b gpu_oob

# Start
main "$@"
