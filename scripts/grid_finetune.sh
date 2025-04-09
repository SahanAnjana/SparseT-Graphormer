#!/bin/bash

# Arrays of hyperparameters to grid search over
models=("graph_pred_mini")
tokens=("--graph_token --cls_token")
blr_values=(1e-3 2e-3 3e-3)
weight_decay_values=(1e-4)
clip_grad_values=(1.0)
dropout_values=(0.1)

declare -A arr_decoder_dim
arr_decoder_dim=(["graph_pred_mini"]=64)

# Loop through all combinations of hyperparameters
for model in "${models[@]}"; do
    for i in "${!tokens[@]}"; do
        for blr in "${blr_values[@]}"; do
            for weight_decay in "${weight_decay_values[@]}"; do
                for clip_grad in "${clip_grad_values[@]}"; do
                    for dropout in "${dropout_values[@]}"; do
		    	              token="${tokens[i]}"
                        decoder_dim="${arr_decoder_dim[$model]}"

                        # Set batch size based on the model
                        if [ "$model" == "graph_pred_mini" ]; then
                            batch_size=4
                        elif [ "$model" == "graph_pred_small" ]; then
                            batch_size=3
			                  elif [ "$model" == "graph_pred_micro" ]; then
                            batch_size=5
                        else
                            echo "Unknown model: $model"
                            exit 1
                        fi

                        # Generate a unique output file for each job to avoid overwriting
                        output_file="/home/rdh1115/scratch/experiments_out/gmae_st/finetune_${model}_blr${blr}_wd${weight_decay}_cg${clip_grad}_${i}.out"

                        # Define output directory based on hyperparameter values
                        output_dir="./finetune_pems_${model}_blr${blr}_wd${weight_decay}_cg${clip_grad}_${i}"

                        # Launch a separate job for each hyperparameter combination
                        sbatch <<EOL
#!/bin/bash
#SBATCH --nodes 2
#SBATCH --gres=gpu:v100l:4          # Request {} GPU "generic resources”.
#SBATCH --tasks-per-node=4   # Request {} process per GPU.
#SBATCH --cpus-per-task=8  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=192000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-10:00:00
#SBATCH --account=def-aevans
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive

#SBATCH --output=${output_file}

module load StdEnv/2020
module load python/3.9.6 cuda cudnn
source gmae_st/bin/activate
cd code/gmae_st/gmae_st

export NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=8

wandb offline

# start training
WANDB__SERVICE_WAIT=300 srun python run_finetune.py \
    --model ${model} \
    --dataset_type traffic \
    --dataset_name pems-bay \
    --n_hist 12 \
    --n_pred 12 \
    --accum_iter 4 \
    --epochs 50 \
    --end_channel ${decoder_dim} \
    --num_workers 8 \
    --batch_size ${batch_size} \
    --output_dir ${output_dir} \
    --pin_mem \
    --fp32 \
    --warmup_epochs 10 \
    --trunc_init \
    --act_fn gelu \
    --use_conv \
    --blr ${blr} \
    --weight_decay ${weight_decay} \
    --clip_grad ${clip_grad} \
    --dropout ${dropout} \
    --layer_decay 0.90 \
    ${token}
deactivate
EOL
                    done
                done
            done
        done
    done
done