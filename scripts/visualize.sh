visualize() {
    # logging
    wandb_usr=unknown

    # dataset
    DATASET_ROOT_DIR=../datasets
    # dataset=modelnet40c_background_5
    # dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c/
    dataset=realsense
    dataset_dir=${DATASET_ROOT_DIR}/GraspNetPointClouds/
    corruption=background
    severity=5

    # classifier
    classifier=DGCNN
    classifier_dir=outputs/dgcnn_modelnet40_best_test.pth

    # diffusion model
    diffusion_dir=outputs/diffusion_model_transformer_modelnet40.npy

    # common hyperparameters
    random_seed=2
    batch_size=16

    # dm hyperparameters
    pre_trans=0
    t_max=0.8
    t_min=0.02
    denoising_thrs=100
    pow=1
    lam_l=1
    lam_h=10
    lr=0.2
    steps=400
    wd=0
    optim=adamax
    optim_end_factor=0.05
    subsample=700
    weighted_reg=True

    python3 adapt.py \
        --t_min ${t_min} \
        --t_max ${t_max} \
        --save_itmd 0 \
        --denoising_thrs ${denoising_thrs} \
        --random_seed ${random_seed} \
        --pow ${pow} \
        --diffusion_steps 500 \
        --diffusion_noise_schedule polynomial_2 \
        --batch_size ${batch_size} \
        --scale_mode unit_std \
        --cls_scale_mode unit_norm \
        --dataset ${dataset} \
        --dataset_dir ${dataset_dir} \
        --classifier ${classifier} \
        --classifier_dir ${classifier_dir} \
        --diffusion_dir ${diffusion_dir} \
        --exp_name ${classifier}_${corruption}_${severity} \
        --mode vis \
        --model transformer \
        --lr ${lr} \
        --n_update ${steps} \
        --weight_decay ${wd} \
        --lam_l ${lam_l} \
        --lam_h ${lam_h} \
        --beta1 0.9 \
        --beta2 0.999 \
        --optim ${optim} \
        --optim_end_factor ${optim_end_factor} \
        --subsample ${subsample} \
        --weighted_reg ${weighted_reg} \
        --wandb_usr ${wandb_usr}
}


visualize
