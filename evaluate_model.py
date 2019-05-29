from ssd.ssdmodel import SSDModel
from trainer.preparedata import PrepareData
from trainer.postprocessingdata import PostProcessingData
from evaluator.evaluator import Evaluator
from experiments import scratch_det18_voc07


if __name__ == '__main__':
    model_params = scratch_det18_voc07.scratch_det_18_params
    params = scratch_det18_voc07.eval_test

    feature_extractor = model_params.feature_extractor
    model_name = model_params.model_name
    weight_decay = model_params.weight_decay
    batch_size = model_params.batch_size
    labels_offset = model_params.labels_offset
    matched_thresholds = model_params.matched_thresholds

    ssd_model = SSDModel(feature_extractor, model_name, weight_decay)
    data_preparer = PrepareData(ssd_model, batch_size, labels_offset, matched_thresholds)
    data_postprocessor = PostProcessingData(ssd_model)
    ssd_evaluator = Evaluator(ssd_model, data_preparer, data_postprocessor, params)

    ssd_evaluator.start_evaluation()