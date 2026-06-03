from yacs.config import CfgNode as CN

_C = CN()

# max number of keyframes
_C.BUFFER_SIZE = 2048*2

# bias patch selection towards high gradient regions?
_C.GRADIENT_BIAS = False
# Select between random, gradient, scorer
_C.PATCH_SELECTOR = "scorer"
# Eval mode of patch selector (random, topk, multinomial)
_C.SCORER_EVAL_MODE = "multi"
_C.SCORER_EVAL_USE_GRID = True
# Normalizer (only evs): norm, standard
_C.NORM = "std"

# VO config (increase for better accuracy)
_C.PATCHES_PER_FRAME = 80
_C.REMOVAL_WINDOW = 20
_C.OPTIMIZATION_WINDOW = 12
_C.PATCH_LIFETIME = 12

# Active-to-frozen edge marginalization. Frozen edges skip corr/update and keep
# contributing their last target and confidence directly to BA.
_C.ACTIVE_EDGE_MARGINALIZATION = False
_C.MARGINALIZE_WEIGHT_THRESH = 0.75
_C.MARGINALIZE_DELTA_THRESH = 0.25
_C.MARGINALIZE_MIN_AGE = 3
_C.MARGINALIZE_CORE_WINDOW = 4

# threshold for keyframe removal
_C.KEYFRAME_INDEX = 4
_C.KEYFRAME_THRESH = 12.5

# camera motion model
_C.MOTION_MODEL = 'DAMPED_LINEAR'
_C.MOTION_DAMPING = 0.5

_C.MIXED_PRECISION = True

cfg = _C
