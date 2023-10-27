# dataset=modelnet40
# dataset_dir=../datasets/modelnet40_ply_hdf5_2048
# dataset=modelnet
# dataset_dir=../datasets/PointDA_data/modelnet
# dataset=shapenet
# dataset_dir=../datasets/PointDA_data/shapenet
# dataset=scannet
# dataset_dir=../datasets/PointDA_data/scannet
dataset=synthetic
dataset_dir=../datasets/GraspNetPointClouds
# dataset=kinect
# dataset_dir=../datasets/GraspNetPointClouds
# dataset=realsense
# dataset_dir=../datasets/GraspNetPointClouds

wandb_usr=drumpt

CUDA_VISIBLE_DEVICES=2,3 python3 train_dm.py \
    --model transformer \
    --dataset ${dataset} \
    --dataset_dir ${dataset_dir} \
    --exp_name ${dataset}_unit_std_random_scale_clsUniformTrue \
    --diffusion_steps 500 \
    --diffusion_noise_schedule polynomial_2 \
    --cls_uniform True \
    --jitter False \
    --scale_mode unit_std \
    --random_scale \
    --no_zero_mean \
    --lr 2e-4 \
    --lr_gamma 0.9995 \
    --batch_size 32 \
    --accum_grad 2 \
    --n_epochs 40000 \
    --test_epochs 400 \
    --num_workers 8 \
    --wandb_usr ${wandb_usr}


# data='scannet'
# scale_mode='unit_val'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
#     --model transformer \
#     --scale_mode ${scale_mode} \
#     --dataset ${data} \
#     --exp_name \
#     ${data}_${scale_mode}_transformer_polynomial_2_500steps_nozeromean_LRExponentialDecay0.9995 \
#     --diffusion_steps 500 \
#     --diffusion_noise_schedule polynomial_2 \
#     --batch_size 64 \
#     --accum_grad 1 \
#     --n_epochs 40000 \
#     --scale 1 \
#     --scale_mode ${scale_mode} \
#     --jitter False \
#     --lr 2e-4 \
#     --lr_gamma 0.9995 \
#     --no_zero_mean \
#     --test_epochs 400 \
#     --cls_uniform True \
#     --num_workers 8 \
#     --wandb_usr drumpt \
#     #test \
#     #--resume \
#     #outputs/${data}_${scale_mode}_transformer_polynomial_2_500steps_nozeromean_LRExponentialDecay0.9995 \


# dataset=scannet
# model=transformer
# lr_gamma=0.9995
# lr=2e-4

# CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
#     --model ${model} \
#     --dataset ${dataset} \
#     --exp_name \
#     ${dataset}_${scale_mode}_transformer_polynomial_2_500steps_${lr}LRExponentialDecay${lr_gamma}_clsUniformFalse \
#     --diffusion_steps 500 \
#     --diffusion_noise_schedule polynomial_2 \
#     --batch_size 16 \
#     --accum_grad 1 \
#     --n_epochs 5000 \
#     --scale_mode unit_val \
#     --lr ${lr} \
#     --lr_gamma ${lr_gamma} \
#     --num_workers 8 \
#     --random_scale \
#     --test_epoch 10 \
#     --wandb_usr drumpt \
#     --no_zero_mean



# # dataset=modelnet
# # model=transformer
# # lr_gamma=0.9995
# # lr=2e-4

# # CUDA_VISIBLE_DEVICES=2,3 python3 main.py \
# #     --model ${model} \
# #     --dataset ${dataset} \
# #     --exp_name \
# #     unit_std_${dataset}_${model}_polynomial_2_500steps_${lr}LRExponentialDecay${lr_gamma}_clsUniformFalse \
# #     --diffusion_steps 500 \
# #     --diffusion_noise_schedule polynomial_2 \
# #     --batch_size 16 \
# #     --accum_grad 1 \
# #     --n_epochs 20000 \
# #     --scale_mode unit_val \
# #     --lr ${lr} \
# #     --lr_gamma ${lr_gamma} \
# #     --num_workers 8 \
# #     --random_scale \
# #     --test_epoch 10 \
# #     --wandb_usr drumpt \
# #     --no_zero_mean


# data='modelnet'
# scale_mode='unit_val'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
#     --model transformer \
#     --scale_mode ${scale_mode} \
#     --dataset ${data} \
#     --exp_name \
#     ${data}_${scale_mode}_transformer_polynomial_2_500steps_nozeromean_LRExponentialDecay0.9995 \
#     --diffusion_steps 500 \
#     --diffusion_noise_schedule polynomial_2 \
#     --batch_size 32 \
#     --accum_grad 1 \
#     --n_epochs 40000 \
#     --scale 1 \
#     --scale_mode ${scale_mode} \
#     --jitter False \
#     --lr 2e-4 \
#     --lr_gamma 0.9995 \
#     --no_zero_mean \
#     --test_epochs 400 \
#     --cls_uniform True \
#     --num_workers 8 \
#     --wandb_usr drumpt \
#     #test \
#     #--resume \
#     #outputs/${data}_${scale_mode}_transformer_polynomial_2_500steps_nozeromean_LRExponentialDecay0.9995 \
