# Finetune v1
# Refer: https://github.com/huggingface/transformers
# base commit id: 1c4a9acc7319221643555c0e8ff1fda2f758c400
# Workpath: transformers/examples/legacy/question-answering

# Dependencies
# $ source python-env/bin/activate
# $ cd $TRAIN_WORK_PATH && mkdir data
# $ wget -O data/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
# $ wget -O data/dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

# source ./python-env/bin/activate
# TRAIN_WORK_PATH=transformers/examples/legacy/question-answering
# cd $TRAIN_WORK_PATH
source ../../../../../python-env/bin/activate

export SQUAD_DIR=`pwd`/data
MODEL_BASE=/home/xiping/mydisk2_2T/mygithub/bugs/mobilebert_jira104177/mobilebert_v1
Model_NV_FP32=$MODEL_BASE/checkpoint-3500
Model_CPU_FP32=$MODEL_BASE/cpu_fp32/checkpoint-4000
Model_CPU_BF16_AMP_SPR=$MODEL_BASE/cpu_bf16_spr/checkpoint-4000

# Test accuracy after training.
python run_squad.py \
  --model_type mobilebert \
  --model_name_or_path $Model_CPU_BF16_AMP_SPR \
  --do_eval \
  --do_lower_case \
  --local_rank -1 \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 64 \
  --learning_rate 4e-5 \
  --num_train_epochs 5.0 \
  --max_seq_length 320 \
  --doc_stride 128 \
  --warmup_steps 1400 \
  --output_dir $SQUAD_DIR/eval 2>&1 | tee eval.log

# All models result:
# ============================================================
# 1: Model_NV_FP32
# {'exact': 81.3907284768212, 'f1': 89.03433915528728, 'total': 10570, 'HasAns_exact': 81.3907284768212, 'HasAns_f1': 89.03433915528728, 'HasAns_total': 10570, 'best_exact': 81.3907284768212, 'best_exact_thresh': 0.0, 'best_f1': 89.03433915528728, 'best_f1_thresh': 0.0}
# 2: Model_CPU_FP32
# {'exact': 81.27719962157049, 'f1': 88.94098600540467, 'total': 10570, 'HasAns_exact': 81.27719962157049, 'HasAns_f1': 88.94098600540467, 'HasAns_total': 10570, 'best_exact': 81.27719962157049, 'best_exact_thresh': 0.0, 'best_f1': 88.94098600540467, 'best_f1_thresh': 0.0}
# 3: Model_CPU_BF16_AMP_SPR
# {'exact': 81.15421002838221, 'f1': 88.99314619286862, 'total': 10570, 'HasAns_exact': 81.15421002838221, 'HasAns_f1': 88.99314619286862, 'HasAns_total': 10570, 'best_exact': 81.15421002838221, 'best_exact_thresh': 0.0, 'best_f1': 88.99314619286862, 'best_f1_thresh': 0.0}