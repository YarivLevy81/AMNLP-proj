python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/squad_dev/SQuAD.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/mrqa_datasets/squad_dev/ \
--tokenizer=t5-base \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--max_feature_length=512 \
--max_number_of_records=1000
