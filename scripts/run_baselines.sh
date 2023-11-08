# logging
wandb_usr=drumpt

# dataset
DATASET_ROOT_DIR=../nfs-client/datasets
CODE_BASE_DIR=../nfs-client/CloudFixer
dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c
adv_attack=False # True, False

# classifier
classifier=DGCNN
classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_modelnet40_best_test.pth

# diffusion model
diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_modelnet40/generative_model_ema_last.npy

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
dda_steps=150
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


run_baselines_modelnet40c() {
    SEED_LIST="2" # "0 1 2"
    METHOD_LIST="tent lame sar pl memo dua bn_stats shot dda"
    CORRUPTION_LIST="background cutout density density_inc distortion distortion_rbf distortion_rbf_inv gaussian impulse lidar occlusion rotation shear uniform upsampling"
    SEVERITY_LIST="5"

    for random_seed in ${SEED_LIST}; do
        for method in ${METHOD_LIST}; do
            for corruption in ${CORRUPTION_LIST}; do
                for severity in ${SEVERITY_LIST}; do # "3 5"
                    dataset=modelnet40c_${corruption}_${severity}
                    exp_name=eval_${classifier}_${dataset}_${method}_${seed}

                    if [ "$method" == "tent" ]; then
                        episodic=False
                        test_optim=AdamW
                        params_to_adapt="LN BN GN"
                        batch_size=64
                        test_lr=1e-4 # 1e-4 1e-3 1e-2
                        num_steps=10 # 1 3 5 10
                    elif [ "$method" == "lame" ]; then
                        episodic=False # placeholder
                        test_optim=AdamW # placeholder
                        test_lr=1e-4 # palceholder
                        params_to_adapt="LN BN GN" # placeholder
                        num_steps=0 # placeholder
                        batch_size=64 # important
                        lame_affinity=kNN # rbf, kNN, linear
                        lame_knn=1 # 1, 3, 5, 10
                        lame_max_steps=10 # 1, 10, 100
                    elif [ "$method" == "sar" ]; then
                        episodic=False
                        test_optim=AdamW
                        params_to_adapt="LN BN GN" # placeholder
                        batch_size=64
                        test_lr=1e-2 # 1e-4 1e-3 1e-2
                        num_steps=3 # 1 3 5 10
                        sar_ent_threshold=0.2 # 0.4, 0.2, 0.6, 0.8
                        sar_eps_threshold=0.1 # 0.01, 0.05, 0.1
                    elif [ "$method" == "sar" ]; then
                        episodic=False
                        test_optim=AdamW
                        params_to_adapt="LN BN GN" # placeholder
                        batch_size=64
                        test_lr=1e-2 # 1e-4 1e-3 1e-2
                        num_steps=3 # 1 3 5 10
                        sar_ent_threshold=0.2 # 0.4, 0.2, 0.6, 0.8
                        sar_eps_threshold=0.1 # 0.01, 0.05, 0.1
                    elif [ "$method" == "pl" ]; then
                        episodic=False
                        test_optim=AdamW
                        params_to_adapt="LN BN GN"
                        batch_size=64
                        test_lr=1e-2 # 1e-4 1e-3 1e-2
                        num_steps=1 # 1 3 5 10
                    elif [ "$method" == "memo" ]; then
                        episodic=True
                        test_optim=AdamW
                        params_to_adapt="all"
                        batch_size=1
                        memo_bn_momentum=1/17
                        test_lr=1e-4 # 1e-6 1e-5 1e-4 1e-3
                        num_steps=2 # "1, 2"
                        memo_num_augs=16 # "16 32 64"
                    elif [ "$method" == "dua" ]; then
                        episodic=False
                        test_optim=AdamW # placeholder
                        params_to_adapt="LN BN GN" # placeholder
                        test_lr=1e-4 # placeholder
                        dua_mom_pre=0.1
                        dua_min_mom=0.005
                        batch_size=64
                        ### hyperparameters to tune for dua
                        num_steps=5 # 1, 3, 5, 10
                        dua_decay_factor=0.9 # 0.9, 0.94, 0.99
                    elif [ "$method" == "bn_stats" ]; then
                        bn_stats_prior=0.2 # 0, 0.2, 0.4, 0.6, 0.8
                    elif [ "$method" == "shot" ]; then
                        test_lr=1e-4 # 1e-4 1e-3 1e-2
                        num_steps=5 # 1 3 5 10
                        shot_pl_loss_weight=0 # 0 0.1, 0.3, 0.5, 1
                    elif [ "$method" == "dda" ]; then
                        batch_size=16
                        episodic=False # placeholder
                        test_optim=AdamW # placeholder
                        params_to_adapt="all" # placeholder
                        num_steps=1 # placeholder
                        test_lr=1e-4 # placeholder
                        dda_steps=150
                        dda_guidance_weight=6
                        dda_lpf_method=fps
                        dda_lpf_scale=4
                    fi

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
    done
}


run_baselines_modelnet40c