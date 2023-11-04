hparam_tune() {
    # dataset
    DATASET_ROOT_DIR=../datasets
    # corruption=occlusion
    # severity=5
    # dataset=modelnet40c_${corruption}_${severity}
    dataset=modelnet40c_original
    dataset_dir=${DATASET_ROOT_DIR}/modelnet40_ply_hdf5_2048/
    adv_attack=False # True, False

    # classifier
    classifier=DGCNN
    classifier_dir=outputs/dgcnn_modelnet40_best_test.pth

    # diffusion model
    diffusion_dir=outputs/diffusion_model_transformer_modelnet40/generative_model_ema_last.npy

    # logging
    wandb_usr=drumpt
    exp_name=hparam_search_${classifier}_${corruption}_${severity}
    SEEDS=2 # "0 1 2"

    # tta hyperparameters
    # method & common hyperparameters
    # method=dua
    # batch_size=64

    # hyperparameters to tune for tent
    method=tent
    episodic=False
    test_optim=AdamW
    params_to_adapt="LN BN GN"
    batch_size=64
    ### hyperparameters to tune for tent
    test_lr=1e-4 # 1e-4 1e-3 1e-2
    num_steps=10 # 1 3 5 10

    # hyperparameters for lame
    # method=lame
    # episodic=False # placeholder
    # test_optim=AdamW # placeholder
    # test_lr=1e-4 # palceholder
    # params_to_adapt="LN BN GN" # placeholder
    # num_steps=0 # meaningless
    # batch_size=16 # important
    # ### hyperparameters to tune for lame
    affinity=rbf # rbf, kNN, linear
    lame_knn=5 # 1, 3, 5

    # hyperparameters for memo
    # method=memo
    # episodic=False
    # test_optim=AdamW
    # params_to_adapt="all"
    # batch_size=1
    # ### hyperparameters to tune for memo
    # test_lr=1e-4 # 1e-6 1e-5 1e-4 1e-3
    # num_steps=1 # "1, 2"
    # num_augs=4 # "4 8 16"

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
            --affinity ${affinity} \
            --lame_knn ${lame_knn} \
            --exp_name ${exp_name} \
            --mode hparam_tune \
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
    # CORRUPTION_LIST="background cutout density density_inc distortion distortion_rbf distortion_rbf_inv gaussian impulse lidar occlusion rotation shear uniform upsampling"
    CORRUPTION_LIST="density_inc distortion_rbf distortion_rbf_inv"
    SEVERITY_LIST=5

    # classifier
    classifier=DGCNN
    classifier_dir=outputs/dgcnn_modelnet40_best_test.pth

    # diffusion model
    diffusion_dir=outputs/diffusion_model_transformer_modelnet40/generative_model_ema_last.npy

    # logging
    wandb_usr=drumpt
    SEEDS=2 # "0 1 2"

    # tta hyperparameters
    method=tent
    episodic=False
    test_optim=AdamW
    params_to_adapt="LN BN GN"
    batch_size=64
    test_lr=1e-4 # 1e-4 1e-3 1e-2
    num_steps=10 # 1 3 5 10
    affinity=rbf # rbf, kNN, linear
    lame_knn=5 # 1, 3, 5

    # dm hyperparameters
    # method=pre_trans
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
                    --exp_name modelnet40c_${corruption}_${severity}_${classifier}_${corruption}_${severity}_${random_seed}_${method} \
                    --episodic ${episodic} \
                    --test_optim ${test_optim} \
                    --params_to_adapt ${params_to_adapt} \
                    --test_lr ${test_lr} \
                    --num_steps ${num_steps} \
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


# adapt_test
# hparam_tune
adapt_modelnet40_c