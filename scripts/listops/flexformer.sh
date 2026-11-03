CUDA_VISIBLE_DEVICES=0
# CURRENT_TIME=$(date +"%m%d_%H%M")
model=flexformer
data=listops

if [ ! -d "./logs/$data/$model" ]; then
    mkdir -p ./logs/$data/$model
fi

bs=8
lr=1e-2
train_steps=50000
eval_freq=500
data_dir=dataset/$data/
embed_dim=256
num_layers=4
qkv_dim=256
mlp_dim=512
num_freq=32
wd=0
model_dir=$data/$model/$bs'_'$lr'_'$num_layers'_'$qkv_dim'_'$mlp_dim'_'$num_freq'_'$wd

nohup python -u $data'_train.py' \
  --model_name $model \
  --batch_size $bs \
  --data_dir $data_dir \
  --learning_rate $lr \
  --weight_decay $wd \
  --num_train_steps $train_steps \
  --eval_frequency $eval_freq \
  --emb_dim $embed_dim \
  --qkv_dim $qkv_dim \
  --mlp_dim $mlp_dim \
  --num_freq $num_freq \
  --model_dir $model_dir > ./logs/$model_dir 2>&1 &