#!/bin/bash

# tmp dir for matplotlib
export MPLCONFIGDIR="/tmp/matplotlib-$USER"
mkdir -p "$MPLCONFIGDIR"

export PYTHONPATH="/sac/src:$PYTHONPATH"

cd /sac/src

python 07_analyze_hyperparameter_distribution.py \
    --study_dir "trials/250704-0736__pd_transformer_encoder__transformer__L2_added__VERY_GOOD/250705-0505__pd_transformer_encoder__transformer__L2_added__HPO_outer_3"
#     --study_dir "trials/250705-1754__rbd_transformer_encoder__transformer__L2_added__VERY_GOOD/250705-1754__rbd_transformer_encoder__transformer__L2_added__HPO_outer_0"
# python 99_count.py
exit
# ---------------------
# HPO for PD with Transformer Encoder
# ---------------------
# for i in {0..4}
# do
#   echo "Running HPO for Outer-fold: $i"
#   python 01_optimize_hpo.py \
#     --condition RBD \
#     --aggregator transformer_encoder \
#     --fold_index $i \
#     --n_trials 300
#   sleep 300 # cool-down
# done
# exit

# ---------------------
# run analysis scripts (commented out)
# ---------------------
# python 04_analyze_best_trials.py \
# 	--base_dir "trials/250705-1754__rbd_transformer_encoder__transformer__L2_added__VERY_GOOD" \
# 	--top_n 100 \
#     --param_filter "weight_decay:>:0.00"
# exit

python 05_analyze_all_outer_folds.py \
    --base_dir "trials/250705-1754__rbd_transformer_encoder__transformer__L2_added__VERY_GOOD" \
    --top_n 100


# trials/250705-1754__rbd_transformer_encoder__transformer__L2_added__VERY_GOOD
# trials/250704-0736__pd_transformer_encoder__transformer__L2_added__VERY_GOOD
