# textbookqa first 3 splits 32:
python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-42-num-examples-32.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/textbookqa1024/42-num-examples-32/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=1024

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-43-num-examples-32.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/textbookqa1024/43-num-examples-32/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=1024

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-44-num-examples-32.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/textbookqa1024/44-num-examples-32/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=1024

# textbookqa first 3 splits 128:
python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-42-num-examples-128.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/textbookqa1024/42-num-examples-128/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=1024

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-43-num-examples-128.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/textbookqa1024/43-num-examples-128/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=1024

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-44-num-examples-128.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/textbookqa1024/44-num-examples-128/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=1024

# textbookqa first 3 splits 512:
python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-42-num-examples-512.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/textbookqa1024/42-num-examples-512/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=1024

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-43-num-examples-512.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/textbookqa1024/43-num-examples-512/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=1024

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/textbookqa-train-seed-44-num-examples-512.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/textbookqa1024/44-num-examples-512/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=1024

# dev:
python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/textbookqa/dev.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/textbookqa1024/dev/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=1024 \
--max_number_of_examples=1000
