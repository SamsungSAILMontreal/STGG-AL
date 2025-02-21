## Bash script to do active-learning
# It auto saves checkpoints and save a file to keep track of which step it is currently on (it will auto-resume)

# What you must do: replace the directories with yours

# Notes:
# --max_epochs for the finetune model must have the same number in exp_xtb_finetune_f_osc and the training_args, otherwise it will return an error
# if you run into bugs, you can do rm ${main_dir}${status} & rm ${mols} to delete the progression files, but do not touch those unless needed because they store the progression for the auto-resume
# It will crash if you dont have a large --num_samples in the generation and the constraints remove those few samples (leading to 0 samples)
# if ${main_dir}${status} does not exists, it will return an error, please ignore, this is normal
# There is a ton of log output, sorry about that...

module load python/3.10
module load cuda/11.8

export PATH=$PATH:CHANGE_TO_YOUR_DIR/xtb-dist/bin
export XTB4STDAHOME=CHANGE_TO_YOUR_DIR/xtb4stda
export PATH=$PATH:$XTB4STDAHOME/exe

TORCH_NCCL_BLOCKING_WAIT=0 CUDA_VISIBLE_DEVICES=0 python train_condgenerator.py --dataset_name xtb --num_layers 3 --tag exp_xtb_finetune_IR_f_osc --bf16 \
--check_sample_every_n_epoch 999 --save_every_n_epoch 1 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 --lambda_predict_prop_always \
--batch_size 128 --lr 2.5e-4 --max_epochs 100 --n_gpu 1 --randomize_order --start_random --scaling_type minmax --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 700 --gpt --no_bias --rmsnorm --rotary --no_test_step --limited_properties --xtb_reward_type IR_f_osc \
--load_checkpoint_path CHANGE_TO_YOUR_DIR/AutoregressiveMolecules_checkpoints/exp_xtb_pretrain_5epoch/last.ckpt

# Checkpoints to use (Can be modified)
ckpt_pretrain="exp_xtb_pretrain_5epoch"
ckpt_finetune="exp_xtb_finetune_IR_f_osc"

# Hyperparameters (Can be modified)
tag="active_learning_IR_fosc"
online="True" # wether to use the latest checkpoint (Online) or always go back to the pretrain model (Offline)
n_active_learning_steps="40" # make as big as desired (if you are not satisifed at the end of the script, you can just increase the number and it will continue from the last checkpoint)
CUDA_VISIBLE_DEVICES="0"
training_args="--dataset_name xtb --num_layers 3 --bf16 \
--check_sample_every_n_epoch 999 --save_every_n_epoch 1 --dropout 0.0 --warmup_steps 100 \
--lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 --lambda_predict_prop_always \
--batch_size 128 --lr 2.5e-4 --max_epochs 100 --n_gpu 1 --randomize_order --start_random \
--scaling_type minmax --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 700 --gpt --no_bias --rmsnorm --rotary --no_test_set_accuracy --no_test_step \
--limited_properties --xtb_reward_type IR_f_osc"
only_training_args=""
sampling_args="--temperature_min 0.6 --temperature_max 1.1 --guidance_min 0.5 --guidance_max 1.5 \
--last --not_allow_empty_bond \
--max_number_of_atoms 70 --max_ring_size 6 --xtb_max_energy 1.24 --remove_duplicates \
--test --specific_conditioning \
--sample_batch_size 250 --num_samples 2000 --best_out_of_k 1 --specific_mae_props f_osc \
--xtb_reward_type IR_f_osc --top_k_to_add 999999999 --specific_minmax 0 1 --top_k_output 100" # For each specific_values, specify if they must be minimized (-1), maximized (1), or run the L2 MAE (0)
min_reward=0.4
max_reward=1.0
cond_args_min="1.15 ${min_reward}" # note: please copy this further below if using adaptive rewards
cond_args_max="1.25 ${max_reward}" # note: please copy this further below if using adaptive rewards
adaptive_rewards=True # Wether to automatically update the conditioning min/max rewards based on top-100 and top-1 rewards

# Main directories, change once to your directories
main_dir="CHANGE_TO_YOUR_DIR/stgg_outputs/"
ckpt_dir="CHANGE_TO_YOUR_DIR/AutoregressiveMolecules_checkpoints/"
out_dir="outputs/"
out_fig_dir="output_figs/"
mkdir -p $ckpt_dir
mkdir -p $main_dir$out_dir
mkdir -p $main_dir$out_fig_dir

# Variables
prev_input=""
ckpt_prev=""
mols_old=""
ckpt_new="${ckpt_finetune}"
status="${tag}_status"
mols="${tag}_mols"
stage="${tag}_stage"
fig="${tag}_fig"
outfile="${tag}_out"
# rm ${main_dir}${status} & rm ${mols}
if ! grep -q "0" ${main_dir}${status}; then
echo "0 " >> ${main_dir}${status}
fi

# Loop over active learning stages
for i in $(seq 1 ${n_active_learning_steps});
do
echo "Stage-${i}"
if ! grep -q "${i}" ${main_dir}${status}; then
# Train (if not stage 1)
if grep -q "1" ${main_dir}${status}; then
echo "Stage-${i} Training"
TORCH_NCCL_BLOCKING_WAIT=0 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train_condgenerator.py \
${training_args} \
${only_training_args} \
--tag ${ckpt_new} \
--load_checkpoint_path ${ckpt_dir}${ckpt_prev}/last.ckpt \
--load_generated_mols ${main_dir}${mols}
fi
fi
# Generate
if [[ "${i}" != "1" ]]; then
mols_old="${mols_old} ${main_dir}${mols}"
fi
mols="${tag}_mols_stage${i}"
if [[ "${i}" != "1" ]]; then
mols_old_="--load_generated_mols ${mols_old}"
else
mols_old_=""
fi
if ! grep -q "${i}" ${main_dir}${status}; then
echo "Stage-${i} Generate"
TORCH_NCCL_BLOCKING_WAIT=0 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train_condgenerator.py \
${training_args} \
${sampling_args} \
${mols_old_} \
--specific_min ${cond_args_min} --specific_max ${cond_args_max} \
--tag ${ckpt_new} \
--append_generated_mols_to_file ${main_dir}${mols} \
--specific_store_output ${main_dir}${out_dir}${stage}_${i}.csv
if [ -z "${prev_input}" ]; then # if first time
python make_plots_reward.py --inputs ${main_dir}${out_dir}${stage}_${i}.csv \
--output_fig ${main_dir}${out_fig_dir}${fig}.png --output ${main_dir}${out_dir}${outfile}.pkl \
--output_min_reward ${main_dir}${out_dir}${outfile}_minreward --output_max_reward ${main_dir}${out_dir}${outfile}_maxreward
else # if previous input exists
python make_plots_reward.py --prev_input ${prev_input} --inputs ${main_dir}${out_dir}${stage}_${i}.csv \
--output_fig ${main_dir}${out_fig_dir}${fig}.png --output ${main_dir}${out_dir}${outfile}.pkl \
--output_min_reward ${main_dir}${out_dir}${outfile}_minreward --output_max_reward ${main_dir}${out_dir}${outfile}_maxreward
fi
fi
prev_input="${main_dir}${out_dir}${outfile}.pkl"
# Update the conditioning rewards based on top-100 (min) and top-1 (max)
if [[ "${adaptive_rewards}" == "True" ]]; then
min_reward=$(<"${main_dir}${out_dir}${outfile}_minreward")
max_reward=$(<"${main_dir}${out_dir}${outfile}_maxreward")
echo "(Adaptive-Reward Update) min-reward=${min_reward} max-reward=${max_reward}"
cond_args_min="1.15 ${min_reward}"
cond_args_max="1.25 ${max_reward}"
fi
echo "${i} " >> ${main_dir}${status}
if [[ "${online}" == "True" ]]; then
ckpt_prev="${ckpt_new}"
else
ckpt_prev="${ckpt_pretrain}"
fi
ckpt_new="${tag}_${i}"
done

