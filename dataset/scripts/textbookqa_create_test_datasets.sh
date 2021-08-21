# textbookqa first 3 splits 32:
python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-42-num-examples-32.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/mrqa_datasets/textbookqa1200/42-num-examples-32/ \
--tokenizer=t5-base \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--max_feature_length=1200

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-43-num-examples-32.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/mrqa_datasets/textbookqa1200/43-num-examples-32/ \
--tokenizer=t5-base \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--max_feature_length=1200

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-44-num-examples-32.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/mrqa_datasets/textbookqa1200/44-num-examples-32/ \
--tokenizer=t5-base \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--max_feature_length=1200

# textbookqa first 3 splits 128:
python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-42-num-examples-128.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/mrqa_datasets/textbookqa1200/42-num-examples-128/ \
--tokenizer=t5-base \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--max_feature_length=1200

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-43-num-examples-128.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/mrqa_datasets/textbookqa1200/43-num-examples-128/ \
--tokenizer=t5-base \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--max_feature_length=1200

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-44-num-examples-128.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/mrqa_datasets/textbookqa1200/44-num-examples-128/ \
--tokenizer=t5-base \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--max_feature_length=1200

# textbookqa first 3 splits 512:
python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-42-num-examples-512.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/mrqa_datasets/textbookqa1200/42-num-examples-512/ \
--tokenizer=t5-base \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--max_feature_length=1200

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-43-num-examples-512.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/mrqa_datasets/textbookqa1200/43-num-examples-512/ \
--tokenizer=t5-base \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--max_feature_length=1200

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-44-num-examples-512.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/mrqa_datasets/textbookqa1200/44-num-examples-512/ \
--tokenizer=t5-base \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--max_feature_length=1200

