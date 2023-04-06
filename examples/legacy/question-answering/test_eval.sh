# Finetune v1
# Refer: https://github.com/huggingface/transformers
# base commit id: 1c4a9acc7319221643555c0e8ff1fda2f758c400
# Workpath: transformers/examples/legacy/question-answering

# Dependencies
# $ source python-env/bin/activate
# $ cd $TRAIN_WORK_PATH && mkdir data
# $ wget -O data/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
# $ wget -O data/dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

# Model from https://huggingface.co/csarron/mobilebert-uncased-squad-v2/tree/main
# wget -O models/pytorch_model.bin https://huggingface.co/csarron/mobilebert-uncased-squad-v2/resolve/main/pytorch_model.bin
# At least need: config.json, vocab.txt, tokenizer_config.json, pytorch_model.bin

# source ./python-env/bin/activate
# TRAIN_WORK_PATH=transformers/examples/legacy/question-answering
# cd $TRAIN_WORK_PATH
# source ../../../../../python-env/bin/activate
source ./pythin-env/bin/activate

# Setup ENV: https://github.com/xipingyan/transformers_mobilebert/tree/xp/train_mobilebert_v1#installation
# pip install transformers
# pip install torch
# pip install tensorboard

export SQUAD_DIR=`pwd`/data
#=================================================
# MODEL_BASE=/home/xiping/mydisk2_2T/mygithub/bugs/mobilebert_jira104177/mobilebert_v1
# Model_NV_FP32=$MODEL_BASE/checkpoint-3500
# Model_CPU_FP32=$MODEL_BASE/cpu_fp32/checkpoint-7000
# Model_CPU_BF16_AMP_SPR=$MODEL_BASE/cpu_bf16_spr/checkpoint-7000
# model_type="mobilebert"

Model_Base=`pwd`/models
model_type="csarron/mobilebert-uncased-squad-v2"

# Test accuracy after training.
python run_squad.py \
  --model_type $model_type \
  --model_name_or_path $Model_Base \
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

# All models result for training eval
# ============================================================
# 1: Model_NV_FP32
# {'exact': 81.3907284768212, 'f1': 89.03433915528728, 'total': 10570, 'HasAns_exact': 81.3907284768212, 'HasAns_f1': 89.03433915528728, 'HasAns_total': 10570, 'best_exact': 81.3907284768212, 'best_exact_thresh': 0.0, 'best_f1': 89.03433915528728, 'best_f1_thresh': 0.0}
# 2: Model_CPU_FP32
# 4000:{'exact': 81.27719962157049, 'f1': 88.94098600540467, 'total': 10570, 'HasAns_exact': 81.27719962157049, 'HasAns_f1': 88.94098600540467, 'HasAns_total': 10570, 'best_exact': 81.27719962157049, 'best_exact_thresh': 0.0, 'best_f1': 88.94098600540467, 'best_f1_thresh': 0.0}
# 7000:{'exact': 81.91106906338695, 'f1': 89.35168282726977, 'total': 10570, 'HasAns_exact': 81.91106906338695, 'HasAns_f1': 89.35168282726977, 'HasAns_total': 10570, 'best_exact': 81.91106906338695, 'best_exact_thresh': 0.0, 'best_f1': 89.35168282726977, 'best_f1_thresh': 0.0}
# 3: Model_CPU_BF16_AMP_SPR
# 4000:{'exact': 81.15421002838221, 'f1': 88.99314619286862, 'total': 10570, 'HasAns_exact': 81.15421002838221, 'HasAns_f1': 88.99314619286862, 'HasAns_total': 10570, 'best_exact': 81.15421002838221, 'best_exact_thresh': 0.0, 'best_f1': 88.99314619286862, 'best_f1_thresh': 0.0}
# 7000:{'exact': 81.8543046357616, 'f1': 89.35603986455591, 'total': 10570, 'HasAns_exact': 81.8543046357616, 'HasAns_f1': 89.35603986455591, 'HasAns_total': 10570, 'best_exact': 81.8543046357616, 'best_exact_thresh': 0.0, 'best_f1': 89.35603986455591, 'best_f1_thresh': 0.0}

# Convert to OpenVION IR.
# $ pip install openvino_dev[onnx]
# $ mo --input_model model_cpu_fp32.onnx
# $ mo --input_model model_cpu_fp32.onnx --data_type FP32 --output_dir ./model_cpu_fp32

# All models result for OpenVINO inference
# ============================================================
# ICL+Model_CPU_FP32+OpenVINO
# fp32_4000: {"exact_match": 81.35288552507096, "f1": 89.00396748938897}
# fp32_7000: {"exact_match": 81.93945127719962, "f1": 89.36494151224085}
# bf16_4000: {"exact_match": 81.3434247871334, "f1": 88.97913566776882}
# bf16_7000: {"exact_match": 81.95837275307474, "f1": 89.3691193021445}
# ICL+Model_CPU_BF16_AMP_SPR+OpenVINO
# fp32_4000: {"exact_match": 81.22043519394512, "f1": 89.0471594979699}
# fp32_7000: {"exact_match": 81.91106906338695, "f1": 89.38523463698625}
# bf16_4000: {"exact_match": 81.23935666982024, "f1": 89.0301616901271}
# bf16_7000: {"exact_match": 81.84484389782403, "f1": 89.33456674345454}

