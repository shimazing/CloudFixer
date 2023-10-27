random_seed=0
test_epochs=100
wandb_usr=drumpt
DATASET_ROOT_DIR=../nfs-client/datasets


train_dm_modelnet40() {
    dataset=modelnet40
    dataset_dir=${DATASET_ROOT_DIR}/modelnet40_ply_hdf5_2048
    n_epochs=5000

    CUDA_VISIBLE_DEVICES=0,1 python3 train_dm.py \
        --model transformer \
        --dataset ${dataset} \
        --dataset_dir ${dataset_dir} \
        --exp_name ${dataset}_unit_std_random_scale_clsUniformTrue \
        --diffusion_steps 500 \
        --diffusion_noise_schedule polynomial_2 \
        --cls_uniform True \
        --jitter False \
        --batch_size 32 \
        --accum_grad 2 \
        --scale_mode unit_std \
        --no_zero_mean \
        --lr 2e-4 \
        --lr_gamma 0.9995 \
        --num_workers 8 \
        --random_seed ${random_seed} \
        --n_epochs ${n_epochs} \
        --test_epochs ${test_epochs} \
        --wandb_usr ${wandb_usr}
}


train_dm_modelnet() {
    dataset=modelnet
    dataset_dir=${DATASET_ROOT_DIR}/PointDA_data/modelnet
    n_epochs=20000

    CUDA_VISIBLE_DEVICES=0,1 python3 train_dm.py \
        --model transformer \
        --dataset ${dataset} \
        --dataset_dir ${dataset_dir} \
        --exp_name ${dataset}_unit_std_random_scale_clsUniformTrue \
        --diffusion_steps 500 \
        --diffusion_noise_schedule polynomial_2 \
        --cls_uniform True \
        --jitter False \
        --batch_size 32 \
        --accum_grad 2 \
        --scale_mode unit_std \
        --no_zero_mean \
        --lr 2e-4 \
        --lr_gamma 0.9995 \
        --num_workers 8 \
        --random_seed ${random_seed} \
        --n_epochs ${n_epochs} \
        --test_epochs ${test_epochs} \
        --wandb_usr ${wandb_usr}
}


train_dm_shapenet() {
    dataset=shapenet
    dataset_dir=${DATASET_ROOT_DIR}/PointDA_data/shapenet
    n_epochs=5000

    CUDA_VISIBLE_DEVICES=2,3 python3 train_dm.py \
        --model transformer \
        --dataset ${dataset} \
        --dataset_dir ${dataset_dir} \
        --exp_name ${dataset}_unit_std_random_scale_clsUniformTrue \
        --diffusion_steps 500 \
        --diffusion_noise_schedule polynomial_2 \
        --cls_uniform True \
        --jitter False \
        --batch_size 32 \
        --accum_grad 2 \
        --scale_mode unit_std \
        --no_zero_mean \
        --lr 2e-4 \
        --lr_gamma 0.9995 \
        --num_workers 8 \
        --random_seed ${random_seed} \
        --n_epochs ${n_epochs} \
        --test_epochs ${test_epochs} \
        --wandb_usr ${wandb_usr}
}


train_dm_scannet() {
    dataset=scannet
    dataset_dir=${DATASET_ROOT_DIR}/PointDA_data/scannet
    n_epochs=20000

    CUDA_VISIBLE_DEVICES=6,7 python3 train_dm.py \
        --model transformer \
        --dataset ${dataset} \
        --dataset_dir ${dataset_dir} \
        --exp_name ${dataset}_unit_std_random_scale_clsUniformTrue \
        --diffusion_steps 500 \
        --diffusion_noise_schedule polynomial_2 \
        --cls_uniform True \
        --jitter False \
        --batch_size 16 \
        --accum_grad 4 \
        --scale_mode unit_std \
        --no_zero_mean \
        --lr 2e-4 \
        --lr_gamma 0.9995 \
        --num_workers 8 \
        --random_seed ${random_seed} \
        --n_epochs ${n_epochs} \
        --test_epochs ${test_epochs} \
        --wandb_usr ${wandb_usr}
}


train_dm_synthetic() {
    dataset=synthetic
    dataset_dir=${DATASET_ROOT_DIR}/GraspNetPointClouds
    n_epochs=5000

    CUDA_VISIBLE_DEVICES=0,1 python3 train_dm.py \
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
        --num_workers 8 \
        --batch_size 16 \
        --accum_grad 4 \
        --save_model True \
        --random_seed ${random_seed} \
        --n_epochs ${n_epochs} \
        --test_epochs ${test_epochs} \
        --wandb_usr ${wandb_usr}
}


train_dm_kinect() {
    dataset=kinect
    dataset_dir=${DATASET_ROOT_DIR}/GraspNetPointClouds
    n_epochs=10000

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
        --num_workers 8 \
        --batch_size 16 \
        --accum_grad 4 \
        --random_seed ${random_seed} \
        --n_epochs ${n_epochs} \
        --test_epochs ${test_epochs} \
        --wandb_usr ${wandb_usr}
}


train_dm_realsense() {
    dataset=realsense
    dataset_dir=${DATASET_ROOT_DIR}/GraspNetPointClouds
    n_epochs=10000

    CUDA_VISIBLE_DEVICES=4,5 python3 train_dm.py \
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
        --num_workers 8 \
        --batch_size 16 \
        --accum_grad 4 \
        --random_seed ${random_seed} \
        --n_epochs ${n_epochs} \
        --test_epochs ${test_epochs} \
        --wandb_usr ${wandb_usr}
}


# train_dm_modelnet40
# train_dm_modelnet
# train_dm_shapenet
# train_dm_scannet
# train_dm_synthetic
# train_dm_kinect
train_dm_realsense