# start
pip install pyransac3d
git config --global --add safe.directory /workspace/VAE-MAE
export PYTHONPATH=$PYTHONPATH:/workspace/VAE-MAE

# test
cd ~/code/VAE-MAE/tools

python test.py --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml --ckpt /root/pp_multihead_nds5823_updated.pth

no:
python tools/test.py --cfg_file tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml --ckpt /root/pp_multihead_nds5823_updated.pth

# finetune
python train.py \
  --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml \
  --pretrained_model /root/voxel_res_pretrain_waymo_full.pth \
  --extra_tag ft_nuscenes_from_waymo_pretrain \

# 改了一下finetune的逻辑，使得每10轮做一次评估，训练结束保存最best的3个点。
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python train.py \
  --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml \
  --pretrained_model ~/SWW/voxel_res_pretrain_waymo_full.pth \
  --extra_tag ft_nuscenes_from_waymo_pretrain \
  --ckpt_save_interval 1 \
  --eval_epoch_interval 10 \
  --topk_best_ckpt 3 \
  --best_metric NDS \
  --max_ckpt_save_num 3 \
  --num_epochs_to_eval 0

# train next
# 记得去yaml修改总轮数
python train.py --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml --ckpt ~/code/VAE-MAE/output/nuscenes_models/cbgs_pp_multihead/ft_nuscenes_from_waymo_pretrain/ckpt/checkpoint_epoch_20.pth --start_epoch 20

# 改了一下finetune的逻辑，使得每10轮做一次评估，训练结束保存最best的3个点。
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python train.py \
  --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml \
  --pretrained_model ~/SWW/voxel_res_pretrain_waymo_full.pth \
  --extra_tag ft_nuscenes_from_waymo_pretrain \
  --ckpt_save_interval 1 \
  --eval_epoch_interval 10 \
  --topk_best_ckpt 3 \
  --best_metric NDS \
  --max_ckpt_save_num 3 \
  --num_epochs_to_eval 0