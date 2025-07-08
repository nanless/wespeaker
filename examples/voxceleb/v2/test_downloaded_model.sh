#!/bin/bash
. ./path.sh || exit 1

stage=1
stop_stage=3

model_dir=/root/workspace/speaker_verification/mix_adult_kid/exp/voxblink2_samresnet100_ft
test_dataset_name=combined_datasets_test_fixratio
trials="trials_mix_adult_kid"
master_port=12355

. tools/parse_options.sh || exit 1

# 设置模型路径和配置
exp_dir=$model_dir/$test_dataset_name
model_path=$model_dir/avg_model.pt
config=$model_dir/config.yaml
data=/root/workspace/speaker_verification/mix_adult_kid/data/$test_dataset_name  # 你的数据目录
gpus="0,1,2,3" # 根据实际情况设置

echo "stage: $stage, stop_stage: $stop_stage"
echo "model_dir: $model_dir"
echo "test_dataset_name: $test_dataset_name"
echo "exp_dir: $exp_dir"
echo "model_path: $model_path"
echo "config: $config"
echo "data: $data"
echo "gpus: $gpus"
echo "trials: $trials"
echo "master_port: $master_port"

# 提取embedding
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
echo "Extracting embeddings..."
  python wespeaker/bin/extract_embeddings_fromwavscp.py \
    --model_dir $model_dir \
    --data_dir $data \
    --exp_dir $exp_dir \
    --gpus $gpus \
    --port $master_port
fi

# 计算分数
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Scoring and computing metrics..."
  python wespeaker/bin/score.py \
    --exp_dir $exp_dir \
    --eval_scp_path $exp_dir/embeddings/xvector.scp \
    --cal_mean True \
    --cal_mean_dir $exp_dir/embeddings/ \
    $data/$trials
fi

# 计算EER 
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "compute metrics (EER/minDCF) ..."
  scores_dir=${exp_dir}/scores
  for x in $trials; do
    python wespeaker/bin/compute_metrics.py \
        --p_target 0.01 \
        --c_fa 1 \
        --c_miss 1 \
        ${scores_dir}/${x}.score \
        2>&1 | tee -a ${scores_dir}/vox1_cos_result
    
    echo "compute DET curve ..."
    python wespeaker/bin/compute_det.py \
        ${scores_dir}/${x}.score
  done
fi

echo "Done."
