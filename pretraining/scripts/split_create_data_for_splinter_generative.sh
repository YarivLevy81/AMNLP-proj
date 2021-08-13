python3 create_pretraining_data.py \
--input_file=/home/yandex/AMNLP2021/sehaik/wiki_split/file_* \
--output_dir=/home/yandex/AMNLP2021/sehaik/processed_wiki_split_2/ \
--vocab_file=vocabs/bert-cased-vocab.txt \
--do_lower_case=False \
--do_whole_word_mask=False \
--max_seq_length=512 \
--max_label_length=512 \
--num_processes=16 \
--dupe_factor=5 \
--max_span_length=10 \
--max_questions_per_seq=30