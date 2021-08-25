python3 convert_mrqa_to_features.py \
--input_path=/home/yandex/AMNLP2021/sehaik/mrqa-few-shot/squad_dev/SQuAD.jsonl \
--output_path=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/squad_dev/data.pt \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--tensor_length=512 \
--max_number_of_examples=1000

