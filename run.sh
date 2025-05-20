exp=gpt-m-new
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/default_config.yaml train_7B.py \
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train.py \
    --ster_batch_size 4 \
    --batch_size 28 \
    --model_name /modelpath/gpt2-large \
    --use_lora \
    --model_save_dir "${exp}" \
    --log_file log.log \
    --lr 4e-5 \
    --max_unlearn_steps 1000 \
    --save_every 50 \
    --ster_weight 0.5 \
    --anti_weight 0.3 \
    --kl_weight 0.2 \
    --mix_anti \
    2>&1 | tee "log/${exp}.log"


#--model_name /private/model/meta-llama/Meta-Llama-3-8B \
#--model_name /private/model/OpenAI/gpt2-medium
#--model_name /private/home/projects/ToxificationReversal/pretrained_models/gpt2-large \
#bash run.sh 2>&1 | tee haha
