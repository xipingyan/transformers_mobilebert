# Finetune v1
# Refer: https://github.com/huggingface/transformers
# base commit id: 1c4a9acc7319221643555c0e8ff1fda2f758c400
# Workpath: transformers/examples/legacy/question-answering

# Dependencies
# $ source python-env/bin/activate
# $ cd $TRAIN_WORK_PATH && mkdir data
# $ wget -O data/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
# $ wget -O data/dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

source ./python-env/bin/activate
TRAIN_WORK_PATH=transformers/examples/legacy/question-answering
cd $TRAIN_WORK_PATH

export SQUAD_DIR=`pwd`/data

# Fine-tune MobileBERT on SQuAD v1.1 with NVIDIA Tesla V100 32GB GPU
# python run_squad.py \
#   --model_type mobilebert \
#   --model_name_or_path google/mobilebert-uncased \
#   --do_train \
#   --do_eval \
#   --do_lower_case \
#   --train_file $SQUAD_DIR/train-v1.1.json \
#   --predict_file $SQUAD_DIR/dev-v1.1.json \
#   --per_gpu_train_batch_size 64 \
#   --per_gpu_eval_batch_size 64 \
#   --learning_rate 4e-5 \
#   --num_train_epochs 5.0 \
#   --max_seq_length 320 \
#   --doc_stride 128 \
#   --warmup_steps 1400 \
#   --output_dir $SQUAD_DIR/mobilebert-uncased-warmup-squad_v1 2>&1 | tee train-mobilebert-warmup-squad_v1.log

# Fine-tune MobileBERT on SQuAD v1.1 with Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz
ft_model_path=/home/xiping/xiping/bugs/mobilebert_jira104177/transformers/examples/legacy/question-answering/data/mobilebert-uncased-warmup-squad_v1/checkpoint-3500/
python run_squad.py \
  --model_type mobilebert \
  --model_name_or_path $ft_model_path \
  --do_train \
  --do_eval \
  --local_rank -1 \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 64 \
  --learning_rate 4e-5 \
  --num_train_epochs 5.0 \
  --max_seq_length 320 \
  --doc_stride 128 \
  --warmup_steps 1400 \
  --output_dir $SQUAD_DIR/mobilebert-uncased-warmup-squad_v1_cpu 2>&1 | tee train-mobilebert-warmup-squad_v1_cpu.log
