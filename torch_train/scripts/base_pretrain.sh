python3 train.py \
--train_dataset=/home/yandex/AMNLP2021/sehaik/new_unpacked_wiki_data \
--eval_dataset=/home/yandex/AMNLP2021/sehaik/new_unpacked_wiki_data_val \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--config=t5-small \
--num_train_steps=20000000 \
--print_loss_steps=5000 \
--save_checkpoints_steps=5000 \
--train_batch_size=1 \
--eval_batch_size=1 \
--calc_f1=0 \
--skip_eval=0 \
--save_best=1 \
--save_latest=1 \
--save_step=0 \
--num_workers=10 \
--output_dir=/home/yandex/AMNLP2021/sehaik/pretrain_1/ \
--warmup_steps=10000 --gamma=0.9999 \
--schedule_steps=10000 \
--gradient_accumulation_steps=1024 \
--learning_rate=1e-4
