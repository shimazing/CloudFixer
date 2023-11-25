# logging
wandb_usr=drumpt

# dataset
DATASET_ROOT_DIR=../e3_diffusion_for_molecules/data/ #nfs-client/datasets
CODE_BASE_DIR=../nfs-client/CloudFixer
# DATASET_ROOT_DIR=../datasets
# CODE_BASE_DIR=../CloudFixer
dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c
adv_attack=False # True, False
scenario=normal
imb_ratio=1

# classifier
classifier=DGCNN
classifier_dir=../pointcloud_TTA/outputs/dgcnn_modelnet40_best_test.pth
#${CODE_BASE_DIR}/outputs/dgcnn_modelnet40_best_test.pth

# diffusion model
diffusion_dir="../e3_diffusion_for_molecules/outputs/modelnet40_unit_std_transformer_polynomial_2_500steps_nozeromean_LRExponentialDecay0.9995_finetune_resume_resume/generative_model_ema_1300.npy"
#${CODE_BASE_DIR}/outputs/diffusion_model_transformer_modelnet40.npy

GPUS=(0 1 2 3)
NUM_GPUS=4
i=0


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
# cloudfixer
t_min=0.02
t_len=0.2
t_max=0.22
pow=1
lam_l=1
lam_h=10
lr=0.02
steps=100
warmup=0.2
wd=0
optim=adamax
optim_end_factor=0.05
subsample=2048
weighted_reg=True
rotation=0.1
######################################################


run_baselines() {
    num_steps=0 # placeholder
    episodic=False # placeholder
    test_optim=AdamW # placeholder
    test_lr=1e-4 # placeholder
    params_to_adapt="all" # placeholder

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 adapt.py \
        --t_min ${t_min} \
        --t_len ${t_len} \
        --warmup ${warmup} \
        --rotation ${rotation} \
        --t_max ${t_max} \
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
        --method cloudfixer \
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
        --exp_name ${exp_name} \
        --mode ${mode} \
        --model transformer \
        --input_lr ${lr} \
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
        #i=$((i + 1))
}



run_cloudfixer_all_experiments() {
    CLASSIFIER_LIST=(DGCNN)
    SEED_LIST="2"
    BATCH_SIZE_LIST="64"
    METHOD_LIST="cloudfixer"

    scenario=normal
    imb_ratio=1
    CORRUPTION_LIST="background" # cutout density density_inc distortion distortion_rbf distortion_rbf_inv gaussian impulse lidar occlusion rotation shear uniform upsampling"
    SEVERITY_LIST="5"
    for random_seed in ${SEED_LIST}; do
        for batch_size in ${BATCH_SIZE_LIST}; do
            for classifier in ${CLASSIFIER_LIST}; do
                for corruption in ${CORRUPTION_LIST}; do
                    for severity in ${SEVERITY_LIST}; do # "3 5"
                        dataset=modelnet40c_${corruption}_${severity}
                        dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c
                        for method in ${METHOD_LIST}; do
                            if [[ "$corruption" == "upsampling" ]]; then
                                batch_size=16
                            else
                                batch_size=16
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



GPUS=(0 1 2 3)
NUM_GPUS=4

run_cloudfixer_all_experiments
