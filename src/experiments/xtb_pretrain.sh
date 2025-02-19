# Pretrain on STGG+ on XTB

module load python/3.10
module load cuda/11.8

export PATH=$PATH:$SCRATCH/xtb-dist/bin
export XTB4STDAHOME=$SCRATCH/xtb4stda
export PATH=$PATH:$XTB4STDAHOME/exe

python data_preprocessing.py --dataset_name xtb --MAX_LEN 700 --limited_properties

## Stage 1 (using default parameters used in STGG+: https://arxiv.org/abs/2407.09357)
TORCH_NCCL_BLOCKING_WAIT=0 CUDA_VISIBLE_DEVICES=0 python train_condgenerator.py --dataset_name xtb --num_layers 3 --tag exp_xtb_pretrain_5epoch --bf16 \
--check_sample_every_n_epoch 999 --save_every_n_epoch 1 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 --lambda_predict_prop_always \
--batch_size 64 --lr 2.5e-4 --max_epochs 5 --n_gpu 1 --randomize_order --start_random --scaling_type minmax --special_init --nhead 16 --swiglu --expand_scale 2.0 \
--max_len 700 --gpt --no_bias --rmsnorm --rotary --no_test_step --limited_properties
