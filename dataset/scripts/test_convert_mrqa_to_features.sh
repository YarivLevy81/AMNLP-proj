python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/test/squad-train-seed-42-num-examples-16.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/test/convert_out \
--tokenizer=t5-base \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--max_feature_length=512