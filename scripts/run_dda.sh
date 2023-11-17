# logging
wandb_usr=drumpt

# dataset
DATASET_ROOT_DIR=../nfs-client/datasets
CODE_BASE_DIR=../nfs-client/CloudFixer
# DATASET_ROOT_DIR=../datasets
# CODE_BASE_DIR=../CloudFixer
dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c
adv_attack=False # True, False
scenario=normal
imb_ratio=1

# classifier
classifier=DGCNN
classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_modelnet40_best_test.pth

# diffusion model
diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_modelnet40.npy

GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=8
i=0

wait_n() {
  # limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  num_max_jobs=8
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

#################### placeholders ####################
# lame
lame_affinity=rbf
lame_knn=3
lame_max_steps=1
# sar
sar_ent_threshold=0.4
sar_eps_threshold=0.05
# memo
memo_num_augs=64
memo_bn_momentum=1/17
# dua
dua_mom_pre=0.1
dua_min_mom=0.005
dua_decay_factor=0.94
# bn_stats
bn_stats_prior=0
# shot
shot_pl_loss_weight=0.3
# dda
dda_steps=100
dda_guidance_weight=6
dda_lpf_method=fps
dda_lpf_scale=4
# cloudfixer
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
######################################################


run_baselines() {
    if [ "$method" == "dda" ]; then
        episodic=False # placeholder
        test_optim=AdamW # placeholder
        test_lr=1e-4 # placeholder
        params_to_adapt="all" # placeholder
        num_steps=1 # placeholder

        dda_steps=100
        dda_guidance_weight=6
        dda_lpf_method=fps
        dda_lpf_scale=4
    fi

    CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python3 adapt.py \
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
        --scenario ${scenario} \
        --imb_ratio ${imb_ratio} \
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
        --lame_affinity ${lame_affinity} \
        --lame_knn ${lame_knn} \
        --lame_max_steps ${lame_max_steps} \
        --sar_ent_threshold ${sar_ent_threshold} \
        --sar_eps_threshold ${sar_eps_threshold} \
        --memo_bn_momentum ${memo_bn_momentum} \
        --memo_num_augs ${memo_num_augs} \
        --dua_mom_pre ${dua_mom_pre} \
        --dua_min_mom ${dua_min_mom} \
        --dua_decay_factor ${dua_decay_factor} \
        --bn_stats_prior ${bn_stats_prior} \
        --shot_pl_loss_weight ${shot_pl_loss_weight} \
        --dda_steps ${dda_steps} \
        --dda_guidance_weight ${dda_guidance_weight} \
        --dda_lpf_method ${dda_lpf_method} \
        --dda_lpf_scale ${dda_lpf_scale} \
        --exp_name ${exp_name} \
        --mode ${mode} \
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
        --wandb_usr ${wandb_usr} \
        2>&1 &
        i=$((i + 1))
    wait_n
}


run_dda() {
    CLASSIFIER_LIST=(DGCNN)
    SEED_LIST="2"
    BATCH_SIZE_LIST="8"
    METHOD_LIST="dda"

    # CORRUPTION_LIST="upsampling"
    # SEVERITY_LIST="5"
    # for random_seed in ${SEED_LIST}; do
    #     for batch_size in ${BATCH_SIZE_LIST}; do
    #         for classifier in ${CLASSIFIER_LIST}; do
    #             for corruption in ${CORRUPTION_LIST}; do
    #                 for severity in ${SEVERITY_LIST}; do # "3 5"
    #                     dataset=modelnet40c_${corruption}_${severity}
    #                     dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c
    #                     classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_modelnet40_best_test.pth
    #                     diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_modelnet40.npy
    #                     for method in ${METHOD_LIST}; do
    #                         exp_name=eval_classifier_${classifier}_dataset_${dataset}_method_${method}_seed_${random_seed}_batch_size_${batch_size}
    #                         mode=eval
    #                         run_baselines
    #                     done
    #                 done
    #             done
    #         done
    #     done
    # done

    # SOURCE_DOMAIN_LIST=(modelnet modelnet shapenet shapenet scannet scannet)
    # TARGET_DOMAIN_LIST=(shapenet scannet modelnet scannet modelnet shapenet)
    # for random_seed in ${SEED_LIST}; do
    #     for batch_size in ${BATCH_SIZE_LIST}; do
    #         for classifier in ${CLASSIFIER_LIST}; do
    #             for ((idx=0; idx<${#SOURCE_DOMAIN_LIST[@]}; ++idx)); do
    #                 dataset=${TARGET_DOMAIN_LIST[idx]}
    #                 dataset_dir=${DATASET_ROOT_DIR}/PointDA_data/${TARGET_DOMAIN_LIST[idx]}
    #                 classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_${SOURCE_DOMAIN_LIST[idx]}_best_test.pth
    #                 diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_${SOURCE_DOMAIN_LIST[idx]}.npy
    #                 for method in ${METHOD_LIST}; do
    #                     exp_name=eval_classifier_${classifier}_source_${SOURCE_DOMAIN_LIST[idx]}_target_${dataset}_method_${method}_seed_${random_seed}_batch_size_${batch_size}
    #                     mode=eval
    #                     run_baselines
    #                 done
    #             done
    #         done
    #     done
    # done

    # SOURCE_DOMAIN_LIST=(synthetic synthetic kinect realsense)
    # TARGET_DOMAIN_LIST=(kinect realsense realsense kinect)
    # for random_seed in ${SEED_LIST}; do
    #     for batch_size in ${BATCH_SIZE_LIST}; do
    #         for classifier in ${CLASSIFIER_LIST}; do
    #             for ((idx=0; idx<${#SOURCE_DOMAIN_LIST[@]}; ++idx)); do
    #                 dataset=${TARGET_DOMAIN_LIST[idx]}
    #                 dataset_dir=${DATASET_ROOT_DIR}/GraspNetPointClouds
    #                 classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_${SOURCE_DOMAIN_LIST[idx]}_best_test.pth
    #                 diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_${SOURCE_DOMAIN_LIST[idx]}.npy
    #                 for method in ${METHOD_LIST}; do
    #                     exp_name=eval_classifier_${classifier}_source_${SOURCE_DOMAIN_LIST[idx]}_target_${dataset}_method_${method}_seed_${random_seed}_batch_size_${batch_size}
    #                     mode=eval
    #                     run_baselines
    #                 done
    #             done
    #         done
    #     done
    # done

    # scenario=mixed
    # CORRUPTION_LIST="background"
    # SEVERITY_LIST="5"
    # for random_seed in ${SEED_LIST}; do
    #     for batch_size in ${BATCH_SIZE_LIST}; do
    #         for classifier in ${CLASSIFIER_LIST}; do
    #             for corruption in ${CORRUPTION_LIST}; do
    #                 for severity in ${SEVERITY_LIST}; do # "3 5"
    #                     dataset=modelnet40c_${corruption}_${severity}
    #                     dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c
    #                     classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_modelnet40_best_test.pth
    #                     diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_modelnet40.npy
    #                     for method in ${METHOD_LIST}; do
    #                         exp_name=eval_classifier_${classifier}_dataset_${dataset}_method_${method}_seed_${random_seed}_batch_size_${batch_size}_scenario_${scenario}
    #                         mode=eval
    #                         scenario=${scenario}
    #                         run_baselines
    #                     done
    #                 done
    #             done
    #         done
    #     done
    # done

    scenario=label_distribution_shift
    imb_ratio=10
    # CORRUPTION_LIST="background cutout density density_inc distortion distortion_rbf distortion_rbf_inv gaussian impulse lidar occlusion rotation shear uniform"
    CORRUPTION_LIST="upsampling"
    BATCH_SIZE_LIST="32"
    METHOD_LIST="dda"

    # CORRUPTION_LIST="background cutout density density_inc distortion distortion_rbf distortion_rbf_inv gaussian impulse lidar occlusion rotation shear uniform upsampling"
    # CORRUPTION_LIST="background cutout density density_inc distortion distortion_rbf distortion_rbf_inv gaussian impulse lidar occlusion rotation shear uniform"
    # CORRUPTION_LIST="upsampling"
    CORRUPTION_LIST="occlusion lidar gaussian rotation shear distortion_rbf_inv"
    SEVERITY_LIST="5"
    for random_seed in ${SEED_LIST}; do
        for batch_size in ${BATCH_SIZE_LIST}; do
            for classifier in ${CLASSIFIER_LIST}; do
                for corruption in ${CORRUPTION_LIST}; do
                    for severity in ${SEVERITY_LIST}; do # "3 5"
                        dataset=modelnet40c_${corruption}_${severity}
                        dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c
                        classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_modelnet40_best_test.pth
                        diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_modelnet40.npy
                        for method in ${METHOD_LIST}; do
                            exp_name=eval_classifier_${classifier}_dataset_${dataset}_method_${method}_seed_${random_seed}_batch_size_${batch_size}
                            mode=eval
                            run_baselines
                        done
                    done
                done
            done
        done
    done

    # SOURCE_DOMAIN_LIST=(modelnet modelnet shapenet shapenet scannet scannet)
    # TARGET_DOMAIN_LIST=(shapenet scannet modelnet scannet modelnet shapenet)
    SOURCE_DOMAIN_LIST=(modelnet modelnet shapenet scannet scannet)
    TARGET_DOMAIN_LIST=(shapenet scannet modelnet modelnet shapenet)
    for random_seed in ${SEED_LIST}; do
        for batch_size in ${BATCH_SIZE_LIST}; do
            for classifier in ${CLASSIFIER_LIST}; do
                for ((idx=0; idx<${#SOURCE_DOMAIN_LIST[@]}; ++idx)); do
                    dataset=${TARGET_DOMAIN_LIST[idx]}
                    dataset_dir=${DATASET_ROOT_DIR}/PointDA_data/${TARGET_DOMAIN_LIST[idx]}
                    classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_${SOURCE_DOMAIN_LIST[idx]}_best_test.pth
                    diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_${SOURCE_DOMAIN_LIST[idx]}.npy
                    for method in ${METHOD_LIST}; do
                        exp_name=eval_classifier_${classifier}_source_${SOURCE_DOMAIN_LIST[idx]}_target_${dataset}_method_${method}_seed_${random_seed}_batch_size_${batch_size}
                        mode=eval
                        run_baselines
                    done
                done
            done
        done
    done
}


run_dda_2() {
    CLASSIFIER_LIST=(DGCNN)
    SEED_LIST="2"
    BATCH_SIZE_LIST="16"
    METHOD_LIST="dda"

    scenario=mixed
    CORRUPTION_LIST="background"
    SEVERITY_LIST="5"
    for random_seed in ${SEED_LIST}; do
        for batch_size in ${BATCH_SIZE_LIST}; do
            for classifier in ${CLASSIFIER_LIST}; do
                for corruption in ${CORRUPTION_LIST}; do
                    for severity in ${SEVERITY_LIST}; do # "3 5"
                        dataset=modelnet40c_${corruption}_${severity}
                        dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c
                        classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_modelnet40_best_test.pth
                        diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_modelnet40.npy
                        for method in ${METHOD_LIST}; do
                            exp_name=eval_classifier_${classifier}_dataset_${dataset}_method_${method}_seed_${random_seed}_batch_size_${batch_size}_scenario_${scenario}
                            mode=eval
                            scenario=${scenario}
                            run_baselines
                        done
                    done
                done
            done
        done
    done

    scenario=label_distribution_shift
    imb_ratio=10
    # CORRUPTION_LIST="background cutout density density_inc distortion distortion_rbf distortion_rbf_inv gaussian impulse lidar occlusion rotation shear uniform upsampling"
    CORRUPTION_LIST="background cutout density density_inc distortion distortion_rbf distortion_rbf_inv gaussian impulse lidar occlusion rotation shear uniform"
    SEVERITY_LIST="5"
    for random_seed in ${SEED_LIST}; do
        for batch_size in ${BATCH_SIZE_LIST}; do
            for classifier in ${CLASSIFIER_LIST}; do
                for corruption in ${CORRUPTION_LIST}; do
                    for severity in ${SEVERITY_LIST}; do
                        dataset=modelnet40c_${corruption}_${severity}
                        dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c
                        classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_modelnet40_best_test.pth
                        diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_modelnet40.npy
                        for method in ${METHOD_LIST}; do
                            exp_name=eval_classifier_${classifier}_dataset_${dataset}_method_${method}_seed_${random_seed}_batch_size_${batch_size}_scenario_${scenario}_imb_ratio_${imb_ratio}
                            mode=eval
                            scenario=${scenario}
                            run_baselines
                        done
                    done
                done
            done
        done
    done

    # SOURCE_DOMAIN_LIST=(synthetic synthetic kinect realsense)
    # TARGET_DOMAIN_LIST=(kinect realsense realsense kinect)
    SOURCE_DOMAIN_LIST=(synthetic realsense)
    TARGET_DOMAIN_LIST=(kinect kinect)
    for random_seed in ${SEED_LIST}; do
        for batch_size in ${BATCH_SIZE_LIST}; do
            for classifier in ${CLASSIFIER_LIST}; do
                for ((idx=0; idx<${#SOURCE_DOMAIN_LIST[@]}; ++idx)); do
                    dataset=${TARGET_DOMAIN_LIST[idx]}
                    dataset_dir=${DATASET_ROOT_DIR}/GraspNetPointClouds
                    classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_${SOURCE_DOMAIN_LIST[idx]}_best_test.pth
                    diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_${SOURCE_DOMAIN_LIST[idx]}.npy
                    for method in ${METHOD_LIST}; do
                        exp_name=eval_classifier_${classifier}_source_${SOURCE_DOMAIN_LIST[idx]}_target_${dataset}_method_${method}_seed_${random_seed}_batch_size_${batch_size}
                        mode=eval
                        run_baselines
                    done
                done
            done
        done
    done
}

# run_dda
run_dda_2