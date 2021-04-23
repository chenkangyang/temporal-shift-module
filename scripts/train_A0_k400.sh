# You should get TSM_kinetics_RGB_repvggA0_softshift8_blockres_avg_segment8_e50.pth
CUDA_VISIBLE_DEVICES=5,6,7,8 python main.py kinetics RGB \
     --arch repvggA0 \
     --num_segments 8 \
     --gd 20 --lr 5e-3 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 83 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --soft_shift --shift_init_mode=shift \
     --npb \
     --gpus 0 1 2 3
