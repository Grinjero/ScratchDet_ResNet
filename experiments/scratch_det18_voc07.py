"""
Root ResNet 18 base
"""

from trainer.trainer import TrainerParams
from evaluator.evaluator import EvaluatorParams


scratch_det_18_params = TrainerParams(
    feature_extractor='root_resnet_18',
    model_name='scratch_det300',
    fine_tune_fe=False,
    train_dir='out/scratch_det_resnet_18_voc_07/logs',
    checkpoint_path='D:/MachineLearning/checkpoints/VOC_2007/scratch_det_18_voc07',
    ignore_missing_vars=False,
    learning_rate=0.05,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    max_number_of_steps=30000,
    optimizer='momentum',
    weight_decay=0.0005,
    num_epochs_per_decay=1,
    end_learning_rate=0.1,
    batch_size=32,
    log_every_n_steps=100,
    save_interval_secs=30*60,
    save_summaries_secs=30,
    labels_offset=0,
    matched_thresholds=0.5
)

# Evaluator parameters

eval_train = EvaluatorParams(
    checkpoint_path='out/scratch_det_resnet_18_voc_07/logs',
    eval_dir='out/scratch_det_resnet_18_voc_07/logs/eval_train',
    use_finetune=False,
    is_training=False,
    eval_train_dataset=True,
    loop=False,
    which_checkpoint='last',
    step_eval_interval=200
)

eval_test = EvaluatorParams(
    checkpoint_path='out/scratch_det_resnet_18_voc_07/logs',
    eval_dir='out/scratch_det_resnet_18_voc_07/eval_test',
    use_finetune=False,
    is_training=False,
    eval_train_dataset=False,
    loop=False,
    which_checkpoint='last',
    step_eval_interval=200
)