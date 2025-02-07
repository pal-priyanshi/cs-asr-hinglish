#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu-v100-32g
#SBATCH --partition=gpu-debug
#### using dgx-spa to make it faster
#SBATCH --mem=50G 
###SBATCH --cpus-per-task=6
###SBATCH --ntasks=1
#SBATCH -J twer_evalbs8
#SBATCH --time=00:10:00
###SBATCH --begin=now+2hour
#SBATCH --output=/scratch/elec/puhe/p/palp3/MUCS/indicwav2vec_outputs/pd_warmup500_rerun_latest/twer/bs8_eval/eval_transliteratedbs8v2_indicwav2vec_MUCS_warmup500gas1_s300shuff100_%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=priyanshi.pal@aalto.fi
###Lower mem could lead to bus error if we don't have enough ram

module load mamba

source activate /scratch/work/palp3/myenv



python /scratch/elec/puhe/p/palp3/MUCS/eval_script_indicwav2vec.py \
    --overwrite_output_dir="False" \
    --resume_from_checkpoint="True" \
    --train_sampling_rate="16000" \
    --dataset_name="MUCS Hindi-English CS dataset" \
    --train_path="/m/triton/scratch/elec/puhe/p/palp3/MUCS/MUCS_train_test_dataset_dict_v2.json" \
    --eval_path="/m/triton/scratch/elec/puhe/p/palp3/MUCS/MUCS_train_test_dataset_dict_v2.json" \
    --model_name_or_path="/scratch/elec/puhe/p/palp3/MUCS/indicwav2vec_outputs/pd_warmup500_rerun_latest/batchsize8/output/checkpoint_3900" \
    --cache_dir="/scratch/elec/puhe/p/palp3/MUCS/indicwav2vec_outputs/pd_warmup500_rerun_latest/twer/bs8_eval" \
    --report_to='wandb'\
    --metric_for_best_model="wer"\
    --greater_is_better="False"\
    --load_best_model_at_end="True" \
    --run_name='transliterated_bs8_indicw2v_ad0_3_hd_02_featd_0_3_lr6e-4_warmup500_s300_shuff100' \
    --dataset_config_name="MUCS-Hin-Eng" \
    --output_dir="/scratch/elec/puhe/p/palp3/MUCS/indicwav2vec_outputs/pd_warmup500_rerun_latest/twer/bs8_eval/output" \
    --max_steps="15000" \
    --per_device_train_batch_size="8" \
    --per_device_eval_batch_size="8" \
    --learning_rate="6e-4" \
    --warmup_steps="500" \
    --lr_scheduler_type="linear" \
    --evaluation_strategy="steps" \
    --chars_to_ignore \" “ ‘ ” � ’ … \
    --save_steps="300" \
    --eval_steps="100" \
    --logging_steps="1" \
    --save_total_limit="6" \
    --freeze_feature_encoder="True" \
    --gradient_checkpointing="True" \
    --do_train="False" \
    --do_eval="True" \
    --seed="300" \
    --audio_column_name="audio_paths" \
    --text_column_name="transcriptions" \
    --eval_metrics cer wer \
    --fp16="True" \
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
    #    --cache_dir="/scratch/elec/puhe/p/palp3/MUCS/indicwav2vec_outputs/pd_warmup500_rerun_latest/twer" \