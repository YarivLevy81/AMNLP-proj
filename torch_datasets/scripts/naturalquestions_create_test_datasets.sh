# naturalquestions first 3 splits 32:
python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/naturalquestions/naturalquestions-train-seed-42-num-examples-32.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/naturalquestions/42-num-examples-32/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/naturalquestions/naturalquestions-train-seed-43-num-examples-32.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/naturalquestions/43-num-examples-32/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/naturalquestions/naturalquestions-train-seed-44-num-examples-32.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/naturalquestions/44-num-examples-32/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

# naturalquestions first 3 splits 128:
python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/naturalquestions/naturalquestions-train-seed-42-num-examples-128.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/naturalquestions/42-num-examples-128/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/naturalquestions/naturalquestions-train-seed-43-num-examples-128.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/naturalquestions/43-num-examples-128/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/naturalquestions/naturalquestions-train-seed-44-num-examples-128.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/naturalquestions/44-num-examples-128/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

# naturalquestions first 3 splits 512:
python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/naturalquestions/naturalquestions-train-seed-42-num-examples-512.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/naturalquestions/42-num-examples-512/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/naturalquestions/naturalquestions-train-seed-43-num-examples-512.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/naturalquestions/43-num-examples-512/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/naturalquestions/naturalquestions-train-seed-44-num-examples-512.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/naturalquestions/44-num-examples-512/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

# dev:
python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/naturalquestions/dev.jsonl \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/naturalquestions/dev/ \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512 \
--max_number_of_examples=1000

