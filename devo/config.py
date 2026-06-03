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
_C.PATCHES_PER_FRAME = 96
_C.REMOVAL_WINDOW = 18
_C.OPTIMIZATION_WINDOW = 10
_C.PATCH_LIFETIME = 9

# Active-to-frozen edge marginalization. Frozen edges skip corr/update and keep
# contributing their last target and confidence directly to BA.
_C.ACTIVE_EDGE_MARGINALIZATION = True
_C.MARGINALIZE_WEIGHT_THRESH = 0.55
_C.MARGINALIZE_DELTA_THRESH = 0.45
_C.MARGINALIZE_MIN_AGE = 2
_C.MARGINALIZE_CORE_WINDOW = 3
_C.MARGINALIZE_WEIGHT_DECAY = 0.99
_C.MARGINALIZE_MAX_ACTIVE_EDGES = 3000
_C.MARGINALIZE_FORCE_BUDGET = True
_C.MARGINALIZE_FORCE_DELTA_THRESH = 1.0
_C.MARGINALIZE_VALIDATE_FROZEN = True
_C.MARGINALIZE_VALIDATE_INTERVAL = 3
_C.MARGINALIZE_SOFT_FROZEN_RESIDUAL = 2.0
_C.MARGINALIZE_MAX_FROZEN_RESIDUAL = 8.0
_C.MARGINALIZE_MIN_FROZEN_WEIGHT_SCALE = 0.2
_C.MARGINALIZE_MAX_FROZEN_EDGES = 6000
_C.MARGINALIZE_PRINT_STATS = False
_C.BA_ITERATIONS = 6

# threshold for keyframe removal
_C.KEYFRAME_INDEX = 4
_C.KEYFRAME_THRESH = 12.5

# camera motion model
_C.MOTION_MODEL = 'DAMPED_LINEAR'
_C.MOTION_DAMPING = 0.5

_C.MIXED_PRECISION = True

cfg = _C
