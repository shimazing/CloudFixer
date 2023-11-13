wandb_usr=drumpt

# dataset
DATASET_ROOT_DIR=../datasets
CODE_BASE_DIR=.

# dataset=modelnet40c_original
# dataset_dir=${DATASET_ROOT_DIR}/modelnet40_ply_hdf5_2048
dataset=modelnet40c_occlusion_5
dataset_dir=${DATASET_ROOT_DIR}/modelnet40_c
# dataset=modelnet
# dataset_dir=${DATASET_ROOT_DIR}/PointDA_data/modelnet
# dataset=shapenet
# dataset_dir=${DATASET_ROOT_DIR}/PointDA_data/shapenet
# dataset=scannet
# dataset_dir=${DATASET_ROOT_DIR}/PointDA_data/scannet
# dataset=synthetic
# dataset_dir=${DATASET_ROOT_DIR}/GraspNetPointClouds
# dataset=kinect
# dataset_dir=${DATASET_ROOT_DIR}/GraspNetPointClouds
# dataset=realsense
# dataset_dir=${DATASET_ROOT_DIR}/GraspNetPointClouds
adv_attack=False

# classifier
classifier=DGCNN
classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_modelnet40_best_test.pth
# classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_modelnet_best_test.pth
# classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_shapenet_best_test.pth
# classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_scannet_best_test.pth
# classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_synthetic_best_test.pth
# classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_kinect_best_test.pth
# classifier_dir=${CODE_BASE_DIR}/outputs/dgcnn_realsense_best_test.pth

# diffusion model
diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_modelnet40.npy
# diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_modelnet.npy
# diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_shapenet.npy
# diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_scannet.npy
# diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_synthetic.npy
# diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_kinect.npy
# diffusion_dir=${CODE_BASE_DIR}/outputs/diffusion_model_transformer_realsense.npy

############# run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=4
i=0
##############################################

wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local num_max_jobs=8
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
dda_steps=150
dda_guidance_weight=6
dda_lpf_method=fps
dda_lpf_scale=4
# ours
ours_steps=150
ours_guidance_weight=6
ours_lpf_method=fps
ours_lpf_scale=4
# hyperparameters for cloudfixer
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


adapt() {
    # # hyperparameters for ours
    method="placeholder"
    batch_size=16
    episodic=False # placeholder
    test_optim=AdamW # placeholder
    params_to_adapt="all" # placeholder
    num_steps=1 # placeholder
    test_lr=1e-4 # placeholder

    # # hyperparameters for ours
    # method=ours
    # batch_size=64
    # episodic=False # placeholder
    # test_optim=AdamW # placeholder
    # params_to_adapt="all" # placeholder
    # num_steps=1 # placeholder
    # test_lr=1e-4 # placeholder
    # # hyperparameters to tune for ours
    # ours_steps=10
    # ours_guidance_weight=6
    # ours_lpf_method=fps
    # ours_lpf_scale=4

    # logging
    SEED_LIST="2"
    # CORRUPTION_LIST="distortion background shear lidar rotation density_inc"
    CORRUPTION_LIST="density_inc"
    SEVERITY_LIST="5"

    for random_seed in ${SEED_LIST}; do
        for corruption in ${CORRUPTION_LIST}; do
            for severity in ${SEVERITY_LIST}; do # "3 5"
                dataset=modelnet40c_${corruption}_${severity}
                exp_name=eval_${classifier}_${dataset}_${method}_${seed}

                CUDA_VISIBLE_DEVICES=2 python3 adapt.py \
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
                    --ours_steps ${ours_steps} \
                    --ours_guidance_weight ${ours_guidance_weight} \
                    --ours_lpf_method ${ours_lpf_method} \
                    --ours_lpf_scale ${ours_lpf_scale} \
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
                    --wandb_usr ${wandb_usr} \
                    2>&1 &
                    wait_n
                    i=$((i + 1))
            done
        done
    done
}


adapt