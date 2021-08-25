# squad first 3 splits 32:
python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/squad/squad-train-seed-42-num-examples-32.jsonl \
--output_path=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/squad/42-num-examples-32/data.pt \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/squad/squad-train-seed-43-num-examples-32.jsonl \
--output_path=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/squad/43-num-examples-32/data.pt \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/squad/squad-train-seed-44-num-examples-32.jsonl \
--output_path=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/squad/44-num-examples-32/data.pt \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

# squad first 3 splits 128:
python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/squad/squad-train-seed-42-num-examples-128.jsonl \
--output_path=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/squad/42-num-examples-128/data.pt \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/squad/squad-train-seed-43-num-examples-128.jsonl \
--output_path=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/squad/43-num-examples-128/data.pt \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/squad/squad-train-seed-44-num-examples-128.jsonl \
--output_path=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/squad/44-num-examples-128/data.pt \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

# squad first 3 splits 512:
python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/squad/squad-train-seed-42-num-examples-512.jsonl \
--output_path=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/squad/42-num-examples-512/data.pt \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/squad/squad-train-seed-43-num-examples-512.jsonl \
--output_path=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/squad/43-num-examples-512/data.pt \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/squad/squad-train-seed-44-num-examples-512.jsonl \
--output_path=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/squad/44-num-examples-512/data.pt \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512

