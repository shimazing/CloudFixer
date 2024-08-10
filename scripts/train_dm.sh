# logging
wandb_usr=unknown
DATASET_ROOT_DIR=../datasets
OUTPUT_DIR=outputs

test_epochs=100
random_seed=0

train_dm_modelnet40() {
    dataset=modelnet40
    dataset_dir=${DATASET_ROOT_DIR}/modelnet40_ply_hdf5_2048
    n_epochs=5000

    python train_dm.py \
        --model transformer \
        --dataset ${dataset} \
        --dataset_dir ${dataset_dir} \
        --output_dir ${OUTPUT_DIR} \
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
        --no_wandb
}

train_dm_modelnet() {
    dataset=modelnet
    dataset_dir=${DATASET_ROOT_DIR}/PointDA_data/modelnet
    n_epochs=20000

    python train_dm.py \
        --model transformer \
        --dataset ${dataset} \
        --dataset_dir ${dataset_dir} \
        --output_dir ${OUTPUT_DIR} \
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
        --no_wandb
}

train_dm_shapenet() {
    dataset=shapenet
    dataset_dir=${DATASET_ROOT_DIR}/PointDA_data/shapenet
    n_epochs=5000

    python train_dm.py \
        --model transformer \
        --dataset ${dataset} \
        --dataset_dir ${dataset_dir} \
        --output_dir ${OUTPUT_DIR} \
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
        --no_wandb
}

train_dm_scannet() {
    dataset=scannet
    dataset_dir=${DATASET_ROOT_DIR}/PointDA_data/scannet
    n_epochs=20000

    python train_dm.py \
        --model transformer \
        --dataset ${dataset} \
        --dataset_dir ${dataset_dir} \
        --output_dir ${OUTPUT_DIR} \
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
        --no_wandb
}

train_dm_modelnet40
# train_dm_modelnet
# train_dm_shapenet
# train_dm_scannet