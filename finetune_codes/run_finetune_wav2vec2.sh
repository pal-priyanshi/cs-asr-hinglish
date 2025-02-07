#!/bin/bash
#SBATCH --gres=gpu:a100:1
####SBATCH --partition=gpu-a100-80g
#### using dgx-spa to make it faster
#SBATCH --mem=100G 
###SBATCH --cpus-per-task=6
###SBATCH --ntasks=1
#SBATCH -J rerun_bestrun_gas1fp16falsebs16
#SBATCH --time=03:00:00
###SBATCH --begin=now+2hour
#SBATCH --output=/scratch/elec/puhe/p/palp3/MUCS/indicwav2vec_outputs/pd_warmup500_rerun_latest/fp16false/output/gas1_indicwav2vec_MUCS_warmup500_s300shuff100_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=priyanshi.pal@aalto.fi
###Lower mem could lead to bus error if we don't have enough ram

module load mamba

source activate /scratch/work/palp3/myenv



python /scratch/elec/puhe/p/palp3/MUCS/finetune_script_indicw2v_partdata.py \
    --overwrite_output_dir="True" \
    --resume_from_checkpoint="True" \
    --train_sampling_rate="16000" \
    --dataset_name="MUCS Hindi-English CS dataset (tags)" \
    --train_path="/m/triton/scratch/elec/puhe/p/palp3/MUCS/mucs_language_segregated_data/MUCS_cs_tags_v2_train_test.json" \
    --eval_path="/m/triton/scratch/elec/puhe/p/palp3/MUCS/mucs_language_segregated_data/MUCS_cs_tags_v2_train_test.json" \
    --model_name_or_path="/m/triton/scratch/elec/puhe/p/palp3/MUCS/indicwav2vec-hindi" \
    --cache_dir="/scratch/elec/puhe/p/palp3/MUCS/indicwav2vec_outputs/pd_warmup500_rerun_latest/fp16false" \
    --report_to='wandb'\
    --metric_for_best_model="wer"\
    --greater_is_better="False"\
    --load_best_model_at_end="True" \
    --run_name='rerun_bestrun_wgas1fp16false_indicw2v_ad0_3_hd_02_featd_0_3_lr6e-4_warmup500_s300_shuff100' \
    --dataset_config_name="MUCS-Hin-Eng (tags)" \
    --output_dir="/scratch/elec/puhe/p/palp3/MUCS/indicwav2vec_outputs/pd_warmup500_rerun_latest/fp16false/output" \
    --max_steps="15000" \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --learning_rate="6e-4" \
    --warmup_steps="500" \
    --lr_scheduler_type="linear" \
    --evaluation_strategy="steps" \
    --chars_to_ignore \" “ ‘ ” � ’ … ? !  \
    --save_steps="300" \
    --eval_steps="100" \
    --logging_steps="1" \
    --save_total_limit="10" \
    --freeze_feature_encoder="True" \
    --gradient_checkpointing="True" \
    --do_train="True" \
    --do_eval="True" \
    --seed="300" \
    --audio_column_name="audio_paths" \
    --text_column_name="transcriptions" \
    --eval_metrics cer wer \
    --fp16="False" \
    --layerdrop="0.0" \
    --attention_dropout="0.3" \
    --group_by_length="True" \
    --hidden_dropout="0.2" \
    --feat_proj_dropout="0.3" \
    --ctc_loss_reduction="mean" \
    --ctc_zero_infinity="True"\
    --max_grad_norm="1.0"\
    --gradient_accumulation_steps="1" \
    --push_to_hub="True" \
    #removed this-     --length_column_name="input_length" \
    #--model_name_or_path="/m/triton/scratch/elec/puhe/p/palp3/MUCS/indicwav2vec-hindi" \
    #    --num_train_epochs="30" \
    #     --warmup_steps="500" \