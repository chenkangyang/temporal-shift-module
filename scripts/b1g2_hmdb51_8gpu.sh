# You should get TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth ATM项目就是从该模型finetune的
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 python main.py hmdb51 RGB \
     --arch repvggB1g2 \
     --num_segments 8 \
     --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --soft_shift --shift_init_mode=shift \
     --npb \
     --gpus 0 1 2 3 4 5 6 7
