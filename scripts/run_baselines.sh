# logging
wandb_usr=drumpt

# dataset
# DATASET_ROOT_DIR=../nfs-client/datasets
# CODE_BASE_DIR=../nfs-client/CloudFixer
DATASET_ROOT_DIR=../datasets
CODE_BASE_DIR=../CloudFixer
# DATASET_ROOT_DIR=../datasets
# CODE_BASE_DIR=../CloudFixer
dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c
adv_attack=False # True, False
scenario=''
imb_ratio=1

# classifier
classifier=DGCNN
classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_modelnet40_best_test.pth

# diffusion model
diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_modelnet40.npy

############ run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=4
i=0
##############################################

wait_n() {
  # limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local num_max_jobs=1
  echo $num_max_jobs
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


# run_baselines_modelnet40c_best_setting() {
#     SEED_LIST="2" # "0 1 2"
#     # METHOD_LIST="tent lame sar pl memo dua bn_stats shot dda"
#     METHOD_LIST="tent lame sar pl memo dua bn_stats shot"
#     CORRUPTION_LIST="background cutout density density_inc distortion distortion_rbf distortion_rbf_inv gaussian impulse lidar occlusion rotation shear uniform upsampling"
#     SEVERITY_LIST="5"

#     for random_seed in ${SEED_LIST}; do
#         for corruption in ${CORRUPTION_LIST}; do
#             for severity in ${SEVERITY_LIST}; do # "3 5"
#                 for method in ${METHOD_LIST}; do
#                     dataset=modelnet40c_${corruption}_${severity}
#                     exp_name=eval_${classifier}_${dataset}_${method}_${random_seed}

#                     if [ "$method" == "tent" ]; then
#                         episodic=False
#                         test_optim=AdamW
#                         test_lr=1e-4 # 1e-4 1e-3 1e-2
#                         params_to_adapt="LN BN GN"
#                         batch_size=64
#                         num_steps=10 # 1 3 5 10
#                     elif [ "$method" == "lame" ]; then
#                         episodic=False # placeholder
#                         test_optim=AdamW # placeholder
#                         test_lr=1e-4 # palceholder
#                         params_to_adapt="LN BN GN" # placeholder
#                         batch_size=64 # important
#                         num_steps=0 # placeholder

#                         lame_affinity=kNN # rbf, kNN, linear
#                         lame_knn=1 # 1, 3, 5, 10
#                         lame_max_steps=10 # 1, 10, 100
#                     elif [ "$method" == "sar" ]; then
#                         episodic=False
#                         test_optim=AdamW
#                         test_lr=1e-2 # 1e-4 1e-3 1e-2
#                         params_to_adapt="LN BN GN" # placeholder
#                         batch_size=64
#                         num_steps=3 # 1 3 5 10

#                         sar_ent_threshold=0.2 # 0.4, 0.2, 0.6, 0.8
#                         sar_eps_threshold=0.1 # 0.01, 0.05, 0.1
#                     elif [ "$method" == "pl" ]; then
#                         episodic=False
#                         test_optim=AdamW
#                         test_lr=1e-2 # 1e-4 1e-3 1e-2
#                         params_to_adapt="LN BN GN"
#                         batch_size=64
#                         num_steps=1 # 1 3 5 10
#                     elif [ "$method" == "memo" ]; then
#                         episodic=True
#                         test_optim=AdamW
#                         test_lr=1e-4 # 1e-6 1e-5 1e-4 1e-3
#                         params_to_adapt="all"
#                         batch_size=1
#                         num_steps=2 # "1, 2"

#                         memo_bn_momentum=1/17
#                         memo_num_augs=16 # "16 32 64"
#                     elif [ "$method" == "dua" ]; then
#                         episodic=False
#                         test_optim=AdamW # placeholder
#                         test_lr=1e-4 # placeholder
#                         params_to_adapt="LN BN GN" # placeholder
#                         batch_size=64
#                         num_steps=5 # 1, 3, 5, 10

#                         ### hyperparameters to tune for dua
#                         dua_mom_pre=0.1
#                         dua_min_mom=0.005
#                         dua_decay_factor=0.9 # 0.9, 0.94, 0.99
#                     elif [ "$method" == "bn_stats" ]; then
#                         episodic=False
#                         test_optim=AdamW # placeholder
#                         test_lr=1e-4 # placeholder
#                         params_to_adapt="LN BN GN" # placeholder
#                         batch_size=64
#                         num_steps=1 # 1, 3, 5, 10

#                         bn_stats_prior=0.2 # 0, 0.2, 0.4, 0.6, 0.8
#                     elif [ "$method" == "shot" ]; then
#                         episodic=False
#                         test_optim=AdamW
#                         test_lr=1e-4 # 1e-4 1e-3 1e-2
#                         params_to_adapt="all"
#                         batch_size=32
#                         num_steps=5 # 1 3 5 10

#                         shot_pl_loss_weight=0 # 0 0.1, 0.3, 0.5, 1
#                     elif [ "$method" == "dda" ]; then
#                         episodic=False # placeholder
#                         test_optim=AdamW # placeholder
#                         test_lr=1e-4 # placeholder
#                         params_to_adapt="all" # placeholder
#                         batch_size=16
#                         num_steps=1 # placeholder

#                         dda_steps=100
#                         dda_guidance_weight=6
#                         dda_lpf_method=fps
#                         dda_lpf_scale=4
#                     fi

#                     CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python3 adapt.py \
#                         --t_min ${t_min} \
#                         --t_max ${t_max} \
#                         --save_itmd 0 \
#                         --denoising_thrs ${denoising_thrs} \
#                         --random_seed ${random_seed} \
#                         --pow ${pow} \
#                         --diffusion_steps 500 \
#                         --diffusion_noise_schedule polynomial_2 \
#                         --batch_size ${batch_size} \
#                         --scale_mode unit_std \
#                         --cls_scale_mode unit_norm \
#                         --dataset ${dataset} \
#                         --dataset_dir ${dataset_dir} \
#                         --classifier ${classifier} \
#                         --classifier_dir ${classifier_dir} \
#                         --diffusion_dir ${diffusion_dir} \
#                         --method ${method} \
#                         --adv_attack ${adv_attack} \
#                         --episodic ${episodic} \
#                         --test_optim ${test_optim} \
#                         --num_steps ${num_steps} \
#                         --test_lr ${test_lr} \
#                         --params_to_adapt ${params_to_adapt} \
#                         --lame_affinity ${lame_affinity} \
#                         --lame_knn ${lame_knn} \
#                         --lame_max_steps ${lame_max_steps} \
#                         --sar_ent_threshold ${sar_ent_threshold} \
#                         --sar_eps_threshold ${sar_eps_threshold} \
#                         --memo_bn_momentum ${memo_bn_momentum} \
#                         --memo_num_augs ${memo_num_augs} \
#                         --dua_mom_pre ${dua_mom_pre} \
#                         --dua_min_mom ${dua_min_mom} \
#                         --dua_decay_factor ${dua_decay_factor} \
#                         --bn_stats_prior ${bn_stats_prior} \
#                         --shot_pl_loss_weight ${shot_pl_loss_weight} \
#                         --dda_steps ${dda_steps} \
#                         --dda_guidance_weight ${dda_guidance_weight} \
#                         --dda_lpf_method ${dda_lpf_method} \
#                         --dda_lpf_scale ${dda_lpf_scale} \
#                         --exp_name ${exp_name} \
#                         --mode ${mode} \
#                         --model transformer \
#                         --lr ${lr} \
#                         --n_update ${steps} \
#                         --weight_decay ${wd} \
#                         --lam_l ${lam_l} \
#                         --lam_h ${lam_h} \
#                         --beta1 0.9 \
#                         --beta2 0.999 \
#                         --optim ${optim} \
#                         --optim_end_factor ${optim_end_factor} \
#                         --subsample ${subsample} \
#                         --weighted_reg ${weighted_reg} \
#                         --wandb_usr ${wandb_usr} \
#                         2>&1 &
#                     wait_n
#                     i=$((i + 1))
#                 done
#             done
#         done
#     done
# }


run_baselines() {
    if [ "$method" == "tent" ]; then
        episodic=False
        test_optim=AdamW
        test_lr=1e-4 # 1e-4 1e-3 1e-2
        params_to_adapt="LN BN GN"
        num_steps=10 # 1 3 5 10
    elif [ "$method" == "lame" ]; then
        episodic=False # placeholder
        test_optim=AdamW # placeholder
        test_lr=1e-4 # palceholder
        params_to_adapt="LN BN GN" # placeholder
        num_steps=0 # placeholder

        lame_affinity=kNN # rbf, kNN, linear
        lame_knn=1 # 1, 3, 5, 10
        lame_max_steps=10 # 1, 10, 100
    elif [ "$method" == "sar" ]; then
        episodic=False
        test_optim=AdamW
        test_lr=1e-2 # 1e-4 1e-3 1e-2
        params_to_adapt="LN BN GN" # placeholder
        num_steps=3 # 1 3 5 10

        sar_ent_threshold=0.2 # 0.4, 0.2, 0.6, 0.8
        sar_eps_threshold=0.1 # 0.01, 0.05, 0.1
    elif [ "$method" == "pl" ]; then
        episodic=False
        test_optim=AdamW
        test_lr=1e-2 # 1e-4 1e-3 1e-2
        params_to_adapt="LN BN GN"
        num_steps=1 # 1 3 5 10
    elif [ "$method" == "memo" ]; then
        episodic=True
        test_optim=AdamW
        test_lr=1e-4 # 1e-6 1e-5 1e-4 1e-3
        params_to_adapt="all"
        batch_size=1
        num_steps=2 # "1, 2"

        memo_bn_momentum=1/17
        memo_num_augs=16 # "16 32 64"
    elif [ "$method" == "dua" ]; then
        episodic=False
        test_optim=AdamW # placeholder
        test_lr=1e-4 # placeholder
        params_to_adapt="LN BN GN" # placeholder
        num_steps=5 # 1, 3, 5, 10

        ### hyperparameters to tune for dua
        dua_mom_pre=0.1
        dua_min_mom=0.005
        dua_decay_factor=0.9 # 0.9, 0.94, 0.99
    elif [ "$method" == "bn_stats" ]; then
        episodic=False
        test_optim=AdamW # placeholder
        test_lr=1e-4 # placeholder
        params_to_adapt="LN BN GN" # placeholder
        num_steps=1 # 1, 3, 5, 10

        bn_stats_prior=0.2 # 0, 0.2, 0.4, 0.6, 0.8
    elif [ "$method" == "shot" ]; then
        episodic=False
        test_optim=AdamW
        test_lr=1e-4 # 1e-4 1e-3 1e-2
        params_to_adapt="all"
        num_steps=5 # 1 3 5 10

        shot_pl_loss_weight=0 # 0 0.1, 0.3, 0.5, 1
    elif [ "$method" == "dda" ]; then
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

    CUDA_VISIBLE_DEVICES=1,2,3 python3 adapt.py \
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
        2>&1
    wait_n
    i=$((i + 1))
}


hparam_tune_modelnet40c() {
    CLASSIFIER_LIST=(DGCNN)

    SEED_LIST="2"
    # TODO:
    BATCH_SIZE_LIST="1 16" # 64 8 1
    # METHOD_LIST="tent lame sar pl memo dua bn_stats shot dda"
    METHOD_LIST="tent lame sar pl memo dua bn_stats shot"
    for random_seed in ${SEED_LIST}; do
        for batch_size in ${BATCH_SIZE_LIST}; do
            for classifier in ${CLASSIFIER_LIST}; do
                dataset=modelnet40c_original
                dataset_dir=${DATASET_ROOT_DIR}/modelnet40_ply_hdf5_2048/
                classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_modelnet40_best_test.pth
                diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_modelnet40/generative_model_ema_last.npy
                for method in ${METHOD_LIST}; do
                    if [[ "$method" == "dda" ]] && [[ "$batch_size" == "1" || "$batch_size" == "64" ]]; then
                        continue
                    fi
                    exp_name=hparam_tune_classifier_${classifier}_dataset_${dataset}_method_${method}_seed_${random_seed}_batch_size_${batch_size}
                    mode=hparam_tune
                    run_baselines
                done
            done
        done
    done
}


hparam_tune_pointda() { 
    CLASSIFIER_LIST=(DGCNN)

    SEED_LIST="2"
    BATCH_SIZE_LIST="64 8 1"
    SOURCE_DOMAIN_LIST=(modelnet shapenet scannet)
    TARGET_DOMAIN_LIST=(modelnet shapenet scannet)
    # METHOD_LIST="tent lame sar pl memo dua bn_stats shot dda"
    METHOD_LIST="tent lame sar pl memo dua bn_stats shot"
    for random_seed in ${SEED_LIST}; do
        for batch_size in ${BATCH_SIZE_LIST}; do
            for classifier in ${CLASSIFIER_LIST}; do
                for ((idx=0; idx<${#SOURCE_DOMAIN_LIST[@]}; ++idx)); do
                    dataset=${TARGET_DOMAIN_LIST[idx]}
                    dataset_dir=${DATASET_ROOT_DIR}/PointDA_data/${TARGET_DOMAIN_LIST[idx]}
                    classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_${SOURCE_DOMAIN_LIST[idx]}_best_test.pth
                    diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_${SOURCE_DOMAIN_LIST[idx]}/generative_model_ema_last.npy
                    for method in ${METHOD_LIST}; do
                        if [[ "$method" == "dda" ]] && [[ "$batch_size" == "1" || "$batch_size" == "64" ]]; then
                            continue
                        fi
                        exp_name=hparam_tune_classifier_${classifier}_dataset_${dataset}_method_${method}_seed_${random_seed}_batch_size_${batch_size}
                        mode=hparam_tune
                        run_baselines
                    done
                done
            done
        done
    done
}


hparam_tune_graspnet() { 
    CLASSIFIER_LIST=(DGCNN) # (DGCNN PointNet)

    SEED_LIST="2"
    BATCH_SIZE_LIST="64 8 1"
    SOURCE_DOMAIN_LIST=(kinect realsense)
    TARGET_DOMAIN_LIST=(kinect realsense)
    # METHOD_LIST="tent lame sar pl memo dua bn_stats shot dda"
    METHOD_LIST="tent lame sar pl memo dua bn_stats shot"
    for random_seed in ${SEED_LIST}; do
        for batch_size in ${BATCH_SIZE_LIST}; do
            for classifier in ${CLASSIFIER_LIST}; do
                for ((idx=0; idx<${#SOURCE_DOMAIN_LIST[@]}; ++idx)); do
                    dataset=${TARGET_DOMAIN_LIST[idx]}
                    dataset_dir=${DATASET_ROOT_DIR}/GraspNetPointClouds
                    classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_${SOURCE_DOMAIN_LIST[idx]}_best_test.pth
                    diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_${SOURCE_DOMAIN_LIST[idx]}/generative_model_ema_last.npy
                    for method in ${METHOD_LIST}; do
                        if [[ "$method" == "dda" ]] && [[ "$batch_size" == "1" || "$batch_size" == "64" ]]; then
                            continue
                        fi
                        exp_name=hparam_tune_classifier_${classifier}_source_${SOURCE_DOMAIN_LIST[idx]}_target_${dataset}_method_${method}_seed_${random_seed}_batch_size_${batch_size}
                        mode=hparam_tune
                        run_baselines
                    done
                done
            done
        done
    done
}


run_baselines_modelnet40c() {
    CLASSIFIER_LIST=(DGCNN)

    SEED_LIST="2"
    BATCH_SIZE_LIST="64 8 1"
    CORRUPTION_LIST="background cutout density density_inc distortion distortion_rbf distortion_rbf_inv gaussian impulse lidar occlusion rotation shear uniform upsampling"
    SEVERITY_LIST="5"
    METHOD_LIST="tent lame sar pl memo dua bn_stats shot dda"
    for random_seed in ${SEED_LIST}; do
        for batch_size in ${BATCH_SIZE_LIST}; do
            for classifier in ${CLASSIFIER_LIST}; do
                for corruption in ${CORRUPTION_LIST}; do
                    for severity in ${SEVERITY_LIST}; do # "3 5"
                        dataset=modelnet40c_${corruption}_${severity}
                        dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c
                        classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_modelnet40_best_test.pth
                        diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_modelnet40/generative_model_ema_last.npy
                        for method in ${METHOD_LIST}; do
                            if [[ "$method" == "dda" ]] && [[ "$batch_size" == "1" || "$batch_size" == "64" ]]; then
                                continue
                            fi
                            exp_name=eval_classifier_${classifier}_dataset_${dataset}_method_${method}_seed_${random_seed}_batch_size_${batch_size}
                            mode=eval
                            run_baselines
                        done
                    done
                done
            done
        done
    done
}


run_baselines_pointda() { 
    CLASSIFIER_LIST=(DGCNN)

    SEED_LIST="2"
    BATCH_SIZE_LIST="64 8 1"
    SOURCE_DOMAIN_LIST=(modelnet modelnet shapenet shapenet scannet scannet)
    TARGET_DOMAIN_LIST=(shapenet scannet modelnet scannet modelnet shapenet)
    METHOD_LIST="tent lame sar pl memo dua bn_stats shot dda"
    for random_seed in ${SEED_LIST}; do
        for batch_size in ${BATCH_SIZE_LIST}; do
            for classifier in ${CLASSIFIER_LIST}; do
                for ((idx=0; idx<${#SOURCE_DOMAIN_LIST[@]}; ++idx)); do
                    dataset=${TARGET_DOMAIN_LIST[idx]}
                    dataset_dir=${DATASET_ROOT_DIR}/PointDA_data/${TARGET_DOMAIN_LIST[idx]}
                    classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_${SOURCE_DOMAIN_LIST[idx]}_best_test.pth
                    diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_${SOURCE_DOMAIN_LIST[idx]}/generative_model_ema_last.npy
                    for method in ${METHOD_LIST}; do
                        if [[ "$method" == "dda" ]] && [[ "$batch_size" == "1" || "$batch_size" == "64" ]]; then
                            continue
                        fi
                        exp_name=eval_classifier_${classifier}_source_${SOURCE_DOMAIN_LIST[idx]}_target_${dataset}_method_${method}_seed_${random_seed}_batch_size_${batch_size}
                        mode=eval
                        run_baselines
                    done
                done
            done
        done
    done
}


run_baselines_graspnet() { 
    CLASSIFIER_LIST=(DGCNN) # (DGCNN PointNet)

    SEED_LIST="2"
    BATCH_SIZE_LIST="64 8 1"
    SOURCE_DOMAIN_LIST=(synthetic synthetic kinect realsense)
    TARGET_DOMAIN_LIST=(kinect realsense realsense kinect)
    METHOD_LIST="tent lame sar pl memo dua bn_stats shot dda"
    for random_seed in ${SEED_LIST}; do
        for batch_size in ${BATCH_SIZE_LIST}; do
            for classifier in ${CLASSIFIER_LIST}; do
                for ((idx=0; idx<${#SOURCE_DOMAIN_LIST[@]}; ++idx)); do
                    dataset=${TARGET_DOMAIN_LIST[idx]}
                    dataset_dir=${DATASET_ROOT_DIR}/GraspNetPointClouds
                    classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_${SOURCE_DOMAIN_LIST[idx]}_best_test.pth
                    diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_${SOURCE_DOMAIN_LIST[idx]}/generative_model_ema_last.npy
                    for method in ${METHOD_LIST}; do
                        if [[ "$method" == "dda" ]] && [[ "$batch_size" == "1" || "$batch_size" == "64" ]]; then
                            continue
                        fi
                        exp_name=eval_classifier_${classifier}_source_${SOURCE_DOMAIN_LIST[idx]}_target_${dataset}_method_${method}_seed_${random_seed}_batch_size_${batch_size}
                        mode=eval
                        run_baselines
                    done
                done
            done
        done
    done
}


run_baselines_modelnet40c_mixed() {
    CLASSIFIER_LIST=(DGCNN)

    scenario=mixed
    SEED_LIST="2"
    BATCH_SIZE_LIST="64 8"
    CORRUPTION_LIST="background"
    SEVERITY_LIST="5"
    METHOD_LIST="tent lame sar pl memo dua bn_stats shot dda"
    for random_seed in ${SEED_LIST}; do
        for batch_size in ${BATCH_SIZE_LIST}; do
            for classifier in ${CLASSIFIER_LIST}; do
                for corruption in ${CORRUPTION_LIST}; do
                    for severity in ${SEVERITY_LIST}; do # "3 5"
                        dataset=modelnet40c_${corruption}_${severity}
                        dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c
                        classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_modelnet40_best_test.pth
                        diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_modelnet40/generative_model_ema_last.npy
                        for method in ${METHOD_LIST}; do
                            if [[ "$method" == "dda" ]] && [[ "$batch_size" == "1" || "$batch_size" == "64" ]] || [[ "$method" != "dda" && "$batch_size" != "64" ]]; then
                                continue
                            fi
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
}


run_baselines_modelnet40c_temporally_correlated() {
    CLASSIFIER_LIST=(DGCNN)

    scenario=temporally_correlated
    SEED_LIST="2"
    BATCH_SIZE_LIST="64 8"
    CORRUPTION_LIST="background cutout density density_inc distortion distortion_rbf distortion_rbf_inv gaussian impulse lidar occlusion rotation shear uniform upsampling"
    SEVERITY_LIST="5"
    METHOD_LIST="tent lame sar pl memo dua bn_stats shot dda"
    for random_seed in ${SEED_LIST}; do
        for batch_size in ${BATCH_SIZE_LIST}; do
            for classifier in ${CLASSIFIER_LIST}; do
                for corruption in ${CORRUPTION_LIST}; do
                    for severity in ${SEVERITY_LIST}; do # "3 5"
                        dataset=modelnet40c_${corruption}_${severity}
                        dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c
                        classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_modelnet40_best_test.pth
                        diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_modelnet40/generative_model_ema_last.npy
                        for method in ${METHOD_LIST}; do
                            if [[ "$method" == "dda" ]] && [[ "$batch_size" == "1" || "$batch_size" == "64" ]] || [[ "$method" != "dda" && "$batch_size" != "64" ]]; then
                                continue
                            fi
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
}



run_baselines_modelnet40c_label_distribution_shift() {
    CLASSIFIER_LIST=(DGCNN)

    scenario=label_distribution_shift
    imb_ratio=10
    SEED_LIST="2"
    BATCH_SIZE_LIST="64 8"
    CORRUPTION_LIST="background cutout density density_inc distortion distortion_rbf distortion_rbf_inv gaussian impulse lidar occlusion rotation shear uniform upsampling"
    SEVERITY_LIST="5"
    METHOD_LIST="tent lame sar pl memo dua bn_stats shot dda"
    for random_seed in ${SEED_LIST}; do
        for batch_size in ${BATCH_SIZE_LIST}; do
            for classifier in ${CLASSIFIER_LIST}; do
                for corruption in ${CORRUPTION_LIST}; do
                    for severity in ${SEVERITY_LIST}; do # "3 5"
                        dataset=modelnet40c_${corruption}_${severity}
                        dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c
                        classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_modelnet40_best_test.pth
                        diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_modelnet40/generative_model_ema_last.npy
                        for method in ${METHOD_LIST}; do
                            if [[ "$method" == "dda" ]] && [[ "$batch_size" == "1" || "$batch_size" == "64" ]] || [[ "$method" != "dda" && "$batch_size" != "64" ]]; then
                                continue
                            fi
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
}


# hparam_tune_modelnet40c
# hparam_tune_pointda
# hparam_tune_graspnet
# run_baselines_modelnet40c_best_setting
# run_baselines_modelnet40c
# run_baselines_pointda
# run_baselines_graspnet
# run_baselines_modelnet40c_mixed
# run_baselines_modelnet40c_temporally_correlated
run_baselines_modelnet40c_label_distribution_shift