# Convert to ONNX and test accuracy after training.


# source ./python-env/bin/activate
# TRAIN_WORK_PATH=transformers/examples/legacy/question-answering
# cd $TRAIN_WORK_PATH
source ../../../../../python-env/bin/activate

export SQUAD_DIR=`pwd`/data
MODEL_BASE=/home/xiping/mydisk2_2T/mygithub/bugs/mobilebert_jira104177/mobilebert_v1
Model_NV_FP32=$MODEL_BASE/checkpoint-3500
Model_CPU_FP32=$MODEL_BASE/cpu_fp32/checkpoint-4000
Model_CPU_BF16_AMP_SPR=$MODEL_BASE/cpu_bf16_spr/checkpoint-4000

# Convert to ONNX
python run_squad.py \
  --model_type mobilebert \
  --model_name_or_path $Model_CPU_FP32 \
  --do_eval \
  --export_onnx \
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
