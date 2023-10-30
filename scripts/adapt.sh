adapt_modelnet40_c() {
    # logging
    wandb_usr=drumpt

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

    # common hyperparameters
    SEEDS=2 # "0 1 2"
    batch_size=32

    # dm hyperparameters
    pre_trans=1
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

    if [ "1" == "$pre_trans" ]; then
        exp_name=${classifier}_${corruption}_${severity}_lr${lr}_ef${optim_end_factor}_t_${t_max}_${t_min}_${steps}iters_betas_0.9_0.999_wd_0_pow${pow}weighted${weighted_reg}_lam_${lam_h}_${lam_l}_cosaneal_wd${wd}_${optim}_schedule_t_tlb_lr_linearLR_sub${subsample}_wRotation0.02_denoisingThrs${denoising_thrs}_trans${trans}_seed${seed}
        pre_trans=--pre_trans
    else
        exp_name=${classifier}_${corruption}_${severity}
        pre_trans=""
    fi

    for random_seed in ${SEEDS}; do
        for corruption in ${CORRUPTION_LIST}; do
            for severity in ${SEVERITY_LIST}; do
                python3 adapt.py \
                    --t_min ${t_min} \
                    --t_max ${t_max} \
                    --save_itmd 0 \
                    --denoising_thrs ${denoising_thrs} \
                    --random_seed ${random_seed} \
                    ${pre_trans} \
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


adapt_poc() { # for modelnetc_background_5
    wandb_usr=drumpt

    # dataset
    DATASET_ROOT_DIR=../datasets
    dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c/
    CORRUPTION_LIST="background"
    SEVERITY_LIST=5

    # classifier
    classifier=DGCNN
    classifier_dir=outputs/dgcnn_modelnet40_best_test.pth

    # diffusion model
    diffusion_dir=outputs/diffusion_model_transformer_modelnet40/generative_model_ema_last.npy

    # common hyperparameters
    SEEDS=2 # "0 1 2"
    batch_size=16

    # dm hyperparameters
    # pre_trans=1
    pre_trans=0
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

    if [ "1" == "$pre_trans" ]; then
        exp_name=${classifier}_${corruption}_${severity}_lr${lr}_ef${optim_end_factor}_t_${t_max}_${t_min}_${steps}iters_betas_0.9_0.999_wd_0_pow${pow}weighted${weighted_reg}_lam_${lam_h}_${lam_l}_cosaneal_wd${wd}_${optim}_schedule_t_tlb_lr_linearLR_sub${subsample}_wRotation0.02_denoisingThrs${denoising_thrs}_trans${trans}_seed${seed}
        pre_trans=--pre_trans
    else
        exp_name=${classifier}_${corruption}_${severity}
        pre_trans=""
    fi

    for random_seed in ${SEEDS}; do
        for corruption in ${CORRUPTION_LIST}; do
            for severity in ${SEVERITY_LIST}; do
                python3 adapt.py \
                    --t_min ${t_min} \
                    --t_max ${t_max} \
                    --save_itmd 0 \
                    --denoising_thrs ${denoising_thrs} \
                    --random_seed ${random_seed} \
                    ${pre_trans} \
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


# adapt_modelnet40_c
adapt_poc