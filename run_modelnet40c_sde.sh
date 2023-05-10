#lambda_s=50####
corruption=background
array=(0)
for lambda_s in "${array[@]}"
do
CUDA_VISIBLE_DEVICES=1,2,3,0 python3 main_sdedit.py \
    --diffusion_steps 500 \
    --diffusion_noise_schedule polynomial_2 \
    --batch_size 32 \
    --scale 1 \
    --scale_mode unit_std \
    --cls_scale_mode unit_norm \
    --dataset modelnet40c_${corruption}_5 \
    --self_ensemble \
    --K 1 \
    --t 0.4 \
    --t_thrs 0.0 \
    --random_seed 2 \
    --exp_name \
    ${corruption}_5_lr1e-2_matching_t0.9_400iters_betas_0.9_0.999_wd_0_l1_0 \
    --domain_cls '' \
    --no_zero_mean \
    --lambda_s $lambda_s \
    --lambda_ent 0 \
    --lambda_i 0 \
    --temperature 2.5 \
    --bn bn \
    --gn False \
    --classifier \
    'outputs/latent_classifier_onlyOri_modelnet40c_fc_normFalse_gnFalsebn/best.pt' \
    --mode vis \
    --radius 0.5 \
    --model transformer \
    --resume \
    outputs/unit_std_modelnet40_transformer_polynomial_2_500steps_nozeromean_2e-4LRExponentialDecay0.9995_clsUniformFalse/generative_model_ema_last.npy \
    --n_subsample 512 \
    --keep_sub True \
    --jitter False \
    --n_reverse_steps 1 \
    --n_iters_per_update 1 \
    --accum 3 \
    --ddim \
    --lr 1e-2 \
    --n_update 400 \
    --weight_decay 0.0 \
    --matching_t 0.9 \
    --l1 0.0 \
    --beta1 0.9 \
    --beta2 0.999 \
    --optim adamw \
    --subsample 1024 \
    --pre_trans \
    #--activate_mask \
    #--no_wandb \
    #${corruption}_5_lr5e-2_400steps_t0.02-0.42 \
    #outputs/unit_std_modelnet40_transformer_polynomial_2_500steps_nozeromean_2e-4LRExponentialDecay0.9995_clsUniformFalse/generative_model_ema_last.npy \
    #--latent_trans \
    #--time_cond \
    #outputs/unit_std_modelnet_transformer_polynomial_2_500steps_nozeromean_2e-4LRExponentialDecay0.9995/generative_model_ema_last.npy \
    #'outputs/latent_classifier_onlyOri_modelnet_fc_normFalse_gnTrue/best.pt' \
    #--random_trans 1 \
    #'outputs/latent_classifier_0.0_fc_normFalse_shapenet2modelnet_withDM0.4_dmpvd_clTrue256lam1.0_temperature0.1_inputTransFalse_rotationAugTrue_timecondTrue_gnTrue_clf_guidanceFalse100.0_tgtTrainMode2.5_srcRandomRemoveFalse_useOriFalse/best.pt' \
    #'outputs/latent_classifier_0.4_fc_normFalse_shapenet2modelnet_withDMTrue_dmpvd_clTrue256lam1.0_temperature0.1_inputTransFalse_rotationAugTrue_timecondTrue_gnTrue_clf_guidanceFalse100.0_tgtTrainMode2.5_srcRandomRemoveFalse_useOriFalse/last.pt' \
    #outputs/domain_classifier_DGCNN_shape_scan_timecondGN_fullt.pt \
    #'outputs/latent_classifier_0.6_fc_normFalse_shapenet2modelnet_withDMFalse_dmpvd_clTrue256lam1.0_temperature0.1_inputTransFalse_rotationAugTrue_timecondTrue_gnTrue_clf_guidanceFalse100.0_tgtTrainMode2.5_srcRandomRemoveFalse_useOriFalse/last.pt' \
    #outputs/latent_classifier_0.6_fc_normFalse_shapenet2modelnet_withDMFalse_dmpvd_clTrue256lam1.0_temperature0.1_inputTransFalse_rotationAugTrue_timecondTrue_gnTrue_clf_guidanceFalse100.0_tgtTrainMode2.5_srcRandomRemoveFalse_useOriFalse
    #'outputs/SDist_0.4_fc_normFalsebngnFalse_shapenet2modelnet_withDMFalse_dmpvd_timecondTrue_clf_guidanceFalse100.0_augFalseFalseTrueTrueTrue_useOriFalse_epochs100_clFalse1024False_byol_onlySrcTrue_step1/best.pt'\
    #--ilvr 0.5 \
    #'outputs/SDist_0.4_fc_normFalsebngnFalse_shapenet2modelnet_withDMFalse_dmpvd_timecondTrue_clf_guidanceFalse100.0_augFalseFalseTrueTrueTrue_useOriFalse_epochs100_clFalse1024False_byol_step2/best.pt'\
    #outputs/unit_std_transformer_polynomial_2_500steps_nozeromean_LRExponentialDecay0.9995_resume_resume/generative_model_ema_last.npy \
    #'outputs/SDist_0.4_fc_normFalsebngnFalse_shapenet2scannet_withDMFalse_dmpvd_timecondTrue_clf_guidanceFalse100.0_augTrueFalseTrueTrue_useOriFalse_epochs100_clFalse1024False_byol_step2/best.pt'\
    #'outputs/SDist_0.4_fc_normFalsebnbngnFalse_shapenet2modelnet_withDMFalse_dmpvd_timecondTrue_clf_guidanceFalse100.0_srcRandomRemoveFalse_useOriFalse_epochs100_step2/best.pt' \
    #--dpm_solver \
    #--latent_subdist \
    #--egsde \
    #'outputs/SDist_0.4_fc_normFalsebnbngnFalse_shapenet2scannet_withDMFalse_dmpvd_timecondTrue_clf_guidanceFalse100.0_srcRandomRemoveTrue_useOriFalse_epochs100_step2/best.pt' \
    #--entropy_guided \
    #'outputs/SDist_latent_classifier_0.4_fc_normFalsebnbngnFalse_shapenet2modelnet_withDMFalse_dmpvd_clFalse1024lam1_temperature0.1_inputTransFalse_timecondTrue_clf_guidanceFalse100.0_tgtTrainModepseudo_label2.5_srcRandomRemoveFalse_useOriFalse_epochs50_step2/last.pt' \
    #'outputs/latent_classifier_0.4_fc_normFalse_shapenet2scannet_withDMTrue_dmpvd_clTrue256lam2.0_temperature0.1_inputTransFalse_rotationAugTrue_timecondTrue_gnTrue_clf_guidanceTrue100.0_tgtTrainModepseudo_label2.5_srcRandomRemoveFalse_useOriFalse/50.pt' \
    #--entropy_guided \
    #--noise_t0 \

#        'outputs/latent_classifier_0.4_fc_normFalse_shapenet2scannet_withDMTrue_dmpvd_clTrue256lam1.0_temperature0.1_inputTransFalse_rotationAugTrue_timecondTrue_gnTrue_clf_guidanceFalse100.0_tgtTrainModeentropy_minimization2.5_srcRandomRemoveFalse_useOriTrue/50.pt' \
#        'outputs/latent_classifier_0.4_fc_normFalse_shapenet2scannet_withDMTrue_dmpvd_clTrue256lam1.0_temperature0.1_inputTransFalse_rotationAugTrue_timecondTrue_gnTrue_clf_guidanceTrue100.0/40.pt'\
#        'outputs/latent_classifier_0.4_fc_normFalse_shapenet2scannet_withDMTrue_dmpvd_clTrue256lam1.0_temperature0.1_inputTransFalse_rotationAugTrue_timecondTrue_gnTrue/last.pt'\
#            'outputs/latent_classifier_0.4_fc_normFalse_shapenet2scannet_withDMFalse_dmpvd_clTrue256lam1.0_temperature0.1_inputTransTrue_rotationAugTrue/50.pt' \
done


#array=(0 100 200 400 800 1600)
#for lambda_s in "${array[@]}"
#do
#    CUDA_VISIBLE_DEVICES=1,2,3,0 python3 main_sdedit.py \
#        --diffusion_steps 500 \
#        --diffusion_noise_schedule polynomial_2 \
#        --batch_size 32 \
#        --classifier \
#            'outputs/latent_classifier_0.4_fc_normFalse_shapenet2scannet_withDMFalse_dmpvd_clTrue256lam1.0_temperature0.1_inputTransTrue_rotationAugTrue/50.pt' \
#        --input_transform \
#        --scale 1 \
#        --scale_mode unit_std \
#        --cls_scale_mode unit_std \
#        --dataset scannet \
#        --self_ensemble \
#        --K 3 \
#        --t 0.2 \
#        --random_seed 1 \
#        --exp_name input_transform_without_dm_lambda_s_$lambda_s \
#        --domain_cls \
#        outputs/domain_classifier_DGCNN_shape_scan_timecondGN_fullt.pt \
#        --no_zero_mean \
#        --model pvd \
#        --resume \
#        outputs/unit_std_pvd_polynomial_2_500steps_nozeromean_LRExponentialDecay0.9995/generative_model_ema_last.npy \
#        --num_workers 8 \
#        --ddim \
#        --mode eval \
#        --no_wandb \
#        --time_cond \
#        --n_reverse_steps 20 \
#        --egsde \
#        --lambda_s $lambda_s \
#        --lambda_i 0 \
#    #--classifier \
#    #    'outputs/latent_classifier_0.4_fc_normFalse_shapenet2scannet_withDMTrue_dmpvd_clTrue256lam1.0_temperature0.1_inputTransFalse_rotationAugTrue/50.pt' \
#    #    'outputs/latent_classifier_0.4_fc_normFalse_shapenet2scannet_withDMFalse_dmpvd_clTrue256lam1.0_temperature0.1_inputTransFalse_rotationAugTrue/50.pt' \
#    #    'outputs/latent_classifier_0.7_fc_normFalse_shapenet2scannet_withDMFalse_dmpvd_clTrue256lam1.0_temperature0.1_inputTransFalse_rotationAugTrue/50.pt' \
#    #    'outputs/latent_classifier_0.7_fc_normFalse_shapenet2scannet_withDMTrue_dmpvd_clTrue256lam1.0_temperature0.1_inputTransFalse/50.pt' \
#    #--classifier \
#    #    '../GAST_ori/experiments/GAST_scannet_nodup/model.ptdgcnn' \
#    #--preprocess \
#    #--model transformer \
#    #--resume \
#    #outputs/unit_std_transformer_polynomial_2_500steps_resume/generative_model_ema_last.npy \
#    #--classifier \
#    #    '../GAST_ori/experiments/GAST_SPST_modelnet/model.ptdgcnn' \
#    #--voxelization \
#    #--voxel_resolution 5 \
#    #--model pvd \
#    #--resume \
#    #outputs/unit_std_pvd_polynomial_2_500steps_resume_resume/generative_model_ema_last.npy \
#    #--mode 'eval' \
#    #--ddim \
#    #--n_inversion_steps 50 \
#    #--n_reverse_steps 50 \
#    #--classifier \
#    #    '../GAST_ori/experiments/GAST_SPST_scannet_lr1e-4_wd0_layernorm_bsz16/model.ptdgcnn' \
#    #outputs/domain_classifier_DGCNN_shape_model_timecondGN.pt \
#    #--n_subsample 128 \
#    #--entropy_guided \
#    #--resume \
#    #outputs/unit_std_shapenet_pointnet_wo_localcond_radius_fourth_polynomial_2_500steps/generative_model_ema_last.npy \
#    #--dynamics_config \
#    #pointnet2/exp_configs/mvp_configs/config_standard_attention_fourth_radius.json \
#    #--no_wandb \
#    #--model transformer \
#    #--resume \
#    #outputs/unit_std_transformer_polynomial_2_500steps_resume/generative_model_ema_last.npy \
#    #    '../GAST_ori/experiments/GAST_SPST/model.ptdgcnn' \
#    #--resume \
#    #outputs/unit_val_shapenet_pointnet_scale1_polynomial_2_500steps_my_config/generative_model_ema_last.npy \
#    #--dynamics_config \
#    #pointnet2/exp_configs/mvp_configs/config_standard_attention_my.json \
#    #--resume \
#    #outputs/unit_val_shapenet_pointnet_scale1_polynomial_2_500steps_my_config/generative_model_ema_last.npy \
#    #    '../DefRec_and_PCM/PointDA/experiments/defrec_pcm_shape2modelnet/model.ptdgcnn' \
#    #     '../GAST/experiments/GAST_balanced_unitnorm_randomscale0.2/model.ptdgcnn' \
#    #--keep_sub \
#    #--resume \
#    #outputs/unit_std_shapenet_pointnet_wo_localcond_radius_double_add_feature_level_polynomial_2_500steps_resume/generative_model_ema_last.npy \
#    #--dynamics_config \
#    #pointnet2/exp_configs/mvp_configs/config_standard_attention_larger_radius_add_feature_level.json \
#    #--resume \
#    #outputs/unit_std_shapenet_pointnet_wo_localcond_radius_double_use_radius_dec_polynomial_2_500steps/generative_model_ema_last.npy \
#    #--dynamics_config \
#    #pointnet2/exp_configs/mvp_configs/config_standard_attention_larger_radius_use_radius_dec.json \
#    #--resume \
#    #outputs/unit_std_shapenet_pointnet_wo_localcond_radius_double_downsample_tcondpnet_polynomial_2_500steps_randomscale/generative_model_ema_last.npy \
#    #--dynamics_config \
#    #pointnet2/exp_configs/mvp_configs/config_standard_attention_larger_radius_downsample_tcondpnet.json \
#    #--resume \
#    #outputs/unit_std_shapenet_pointnet_wo_localcond_radius_double_downsample_polynomial_2_500steps/generative_model_ema_last.npy \
#    #--dynamics_config \
#    #pointnet2/exp_configs/mvp_configs/config_standard_attention_larger_radius_downsample.json \
#    #--resume \
#    #outputs/unit_std_shapenet_pointnet_wo_localcond_radiusdouble_balanced_resume/generative_model_ema_last.npy \
#    #--dynamics_config \
#    #pointnet2/exp_configs/mvp_configs/config_standard_attention_larger_radius.json \
#    #--resume \
#    #outputs/unit_std_shapenet_pointnet_wo_localcond_radiusdouble_balanced_linear1000/generative_model_ema_last.npy \
#    #--dynamics_config \
#    #pointnet2/exp_configs/mvp_configs/config_standard_attention_larger_radius.json \
#    #    'best_model_shape_model.pt' \
#    #    '../GAST/experiments/GAST_SPST/model.ptdgcnn' \
#done
