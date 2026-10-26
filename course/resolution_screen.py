"""Checks for resolution ablation; no inference-size gains credited to training."""
import torch
from course.screen_protocol import compare_protocol


def validate_resolution_pair(candidate, reference, recorded_control):
    assert candidate["COURSE"]["TEST_SIZE"]==reference["COURSE"]["TEST_SIZE"]==1280
    assert candidate["INPUT"]["IMAGE_SIZE"]==1280 and reference["INPUT"]["IMAGE_SIZE"]==1024
    assert recorded_control["INPUT"]["IMAGE_SIZE"]==recorded_control["COURSE"]["TEST_SIZE"]==1024
    inference_change=compare_protocol(reference,recorded_control,["COURSE.TEST_SIZE"])
    training_change=compare_protocol(candidate,reference,["INPUT.IMAGE_SIZE"])
    assert set(inference_change)=={"COURSE.TEST_SIZE"}
    assert set(training_change)=={"INPUT.IMAGE_SIZE"}
    # compare_protocol intentionally excludes opt-in architecture descriptors;
    # this experiment must keep them IDENTICAL, including all default modules.
    for section in ("QUERY","LOCAL","POINT","SCREEN"):
        defaults={"POINT":{"MODE":"off","CHANNELS":16,"HIDDEN":64,"INFERENCE_POINTS":2048,"QUERY_CHUNK":16,"GAIN":1.,"LR_MULTIPLIER":10.},
                  "LOCAL":{"MODE":"off","CHANNELS":32,"LR_MULTIPLIER":10.}}
        ref=recorded_control["COURSE"].get(section,defaults.get(section))
        assert candidate["COURSE"].get(section)==reference["COURSE"].get(section)==ref,section
    for path in (("BACKBONE","NAME"),("SEM_SEG_HEAD","NAME"),("MASK_FORMER","TRANSFORMER_DECODER_NAME")):
        assert candidate["MODEL"][path[0]][path[1]]==reference["MODEL"][path[0]][path[1]]==recorded_control["MODEL"][path[0]][path[1]]
    return dict(training_difference=training_change,common_inference_difference_from_old_report=inference_change)


def assert_same_initial_state(a,b):
    assert set(a)==set(b),"Initial model tensor sets differ"
    assert getattr(a,"_metadata",None)==getattr(b,"_metadata",None),"Checkpoint version metadata differs"
    mismatch=[k for k,v in a.items() if not torch.equal(v,b[k])]
    assert not mismatch,mismatch
    return len(a)


def paired_scores(candidate,reference):
    assert [v['additional_training_step'] for v in candidate]==[0,1000,2000,3000]
    assert [v['additional_training_step'] for v in reference]==[0,1000,2000,3000]
    for a,b in zip(candidate,reference):
        assert a['input_size']==b['input_size']==1280, 'Mixed inference resolutions are not a training comparison'
        assert a['panel_sha256']==b['panel_sha256'] and a['validation_images']==b['validation_images']==200
    assert abs(candidate[0]['mask_aps']-reference[0]['mask_aps'])<1e-10
    return {str(a['additional_training_step']):a['mask_aps']-b['mask_aps'] for a,b in zip(candidate,reference)}
