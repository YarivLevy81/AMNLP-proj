python3 create_pretraining_data.py \
--input_file=/home/yandex/AMNLP2021/sehaik/wiki_split/file_* \
--output_dir=/home/yandex/AMNLP2021/sehaik/torch_processed_wiki_split/ \
--max_questions_per_seq=15 \
--max_seq_length=256 \
--max_feature_length=512 \
--masked_lm_prob=0.15 \
--max_span_length=10 \
--tokenizer=t5-small \
--cache_dir=/home/yandex/AMNLP2021/sehaik/hugginface_cache