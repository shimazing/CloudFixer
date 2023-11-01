adapt_test() {
    # dataset
    DATASET_ROOT_DIR=../datasets
    corruption=occlusion
    severity=5
    dataset=modelnet40c_${corruption}_${severity}
    dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c/
    adv_attack=False # True, False

    # classifier
    classifier=DGCNN
    classifier_dir=outputs/dgcnn_modelnet40_best_test.pth

    # diffusion model
    diffusion_dir=outputs/diffusion_model_transformer_modelnet40/generative_model_ema_last.npy

    # logging
    wandb_usr=drumpt
    exp_name=${classifier}_${corruption}_${severity}

    # method & common hyperparameters
    # method=dua
    SEEDS=2 # "0 1 2"
    batch_size=64
    method=sar

    # tta hyperparameters
    episodic=True
    num_steps=1
    test_optim=AdamW
    test_lr=1e-4
    params_to_adapt="LN BN"

    # hyperparameters for shot
    # method=shot
    # episodic=True
    # num_steps=1
    # test_optim=AdamW
    # test_lr=1e-4
    # params_to_adapt="all"

    # dm hyperparameters
    t_min=0.02
    t_max=0.8
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

    for random_seed in ${SEEDS}; do
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
            --method ${method} \
            --adv_attack ${adv_attack} \
            --episodic ${episodic} \
            --test_optim ${test_optim} \
            --num_steps ${num_steps} \
            --test_lr ${test_lr} \
            --params_to_adapt ${params_to_adapt} \
            --exp_name ${exp_name} \
            --mode eval \
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
    done
}


adapt_modelnet40_c() {
    # dataset
    DATASET_ROOT_DIR=../datasets
    dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c/
    CORRUPTION_LIST="background cutout density density_inc distortion distortion_rbf distortion_rbf_inv gaussian impulse lidar occlusion rotation shear uniform upsampling"
    SEVERITY_LIST=5

    # classifier
    classifier=DGCNN
    classifier_dir=outputs/dgcnn_modelnet40_best_test.pth

    # diffusion model
    diffusion_dir=outputs/diffusion_model_transformer_modelnet40/generative_model_ema_last.npy

    # logging
    wandb_usr=drumpt
    exp_name=${classifier}_${corruption}_${severity}

    # common hyperparameters
    SEEDS=2 # "0 1 2"
    batch_size=16

    # dm hyperparameters
    method=pre_trans
    t_min=0.02
    t_max=0.8
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

    for random_seed in ${SEEDS}; do
        for corruption in ${CORRUPTION_LIST}; do
            for severity in ${SEVERITY_LIST}; do
                CUDA_VISIBLE_DEVICES=0,1,2,3 python3 adapt.py \
                    --t_min ${t_min} \
                    --t_max ${t_max} \
                    --save_itmd 0 \
                    --denoising_thrs ${denoising_thrs} \
                    --random_seed ${random_seed} \
                    --method ${method}\
                    --pow ${pow} \
                    --diffusion_steps 500 \
                    --diffusion_noise_schedule polynomial_2 \
                    --batch_size ${batch_size} \
                    --scale_mode unit_std \
                    --cls_scale_mode unit_norm \
                    --dataset modelnet40c_${corruption}_${severity} \
                    --dataset_dir ${dataset_dir} \
                    --classifier ${classifier} \
                    --classifier_dir ${classifier_dir} \
                    --diffusion_dir ${diffusion_dir} \
                    --exp_name ${exp_name} \
                    --mode eval \
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
            done
        done
    done
}


adapt_test
# adapt_modelnet40_c