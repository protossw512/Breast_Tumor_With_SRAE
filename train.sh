CUDA_VISIBLE_DEVICES=0 python train.py \
                    --data_dir=/home/wangxiny/Bio/Model_0412_1 \
                    --ckpt_save_path=/home/wangxiny/Bio/Breast_Tumor_With_SRAE/train_0515_1 \
                    --learning_rate=0.0006 \
                    --learning_rate_decay_rate=0.1 \
                    --weight_decay=0.00005 \
                    --max_epoch=150 \
                    --decay_per_epoch=50
