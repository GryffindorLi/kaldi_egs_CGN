#!/bin/bash

# This script was modified from the Tedlium egs.
# It assumes you first completed the run.sh from the main egs/CGN directory

## how you run this (note: this assumes that the run_tdnn.sh soft link points here;
## otherwise call it directly in its location).
# 
# local/chain/run_tdnn.sh

# This script is uses an xconfig-based mechanism
# to get the configuration.

# set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).

# repurposed from egs/Tedlium for CGN by LvdW
use_gpu_decode=false
remove_egs=false
stage=17
stop_stage=19
align_fmllr_lats_stage=-10
nj=1
decode_nj=1
base_eval=dev_s # or dev_t_16khz
gender_eval=female # or male
num_threads_decode=1
min_seg_len=1.55
xent_regularize=0.025
train_set=train
get_egs_stage=-10
gmm=tri4  # the gmm for the target data
num_threads_ubm=32
nnet3_affix=  # cleanup affix for nnet3 and chain dirs, e.g. _cleaned

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
train_stage=-10
train_exit_stage=
tree_affix=  # affix for tree directory, e.g. "a" or "b", in case we change the configuration.
tdnn_blstm_affix=1a  #affix for TDNN directory, e.g. "a" or "b", in case we change the configuration.
common_egs_dir= #/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/cgn/s5/exp/chain/tdnn_blstm1a_sp_bi_epoch2_ld5/egs

#/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/cgn/s5_vl_only/exp/chain/tdnn_blstm1a_sp_bi_epoch1_ld5/egs
chunk_width=150
chunk_left_context=40
chunk_right_context=40
self_repair_scale=0.00001
label_delay=5
dropout_schedule='0,0@0.20,0.3@0.50,0'
# decode options
extra_left_context=50
extra_right_context=50
frames_per_chunk=150
decode_iter=final #acoustic model used in decoding 
# some settings dependent on the GPU, for a single GTX980Ti these settings seem to work ok.
#
# increase these if you have multiple GPUs
num_jobs_initial=1
num_jobs_final=1
num_epochs=1
# change these for different amounts of memory
#num_chunks_per_minibatc=h"256,128,64"
num_chunks_per_minibatch="128,64"
frames_per_iter=1500000
# resulting in around 2500 iters
# End configuration section.
echo "$0 $@"  # Print the command line for logging

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ "$decode_iter" = "final" ]; then
  decode_iter_suffix=""
else
  decode_iter_suffix="_iter${decode_iter}"
fi

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# local/nnet3/run_ivector_common.sh --stage $stage \
#                                   --nj $nj \
#                                   --min-seg-len $min_seg_len \
#                                   --train-set $train_set \
#                                   --gmm $gmm \
#                                   --num-threads-ubm $num_threads_ubm \
#                                   --nnet3-affix "$nnet3_affix"


gmm_dir=exp/$train_set/$gmm
ali_dir=exp/$train_set/${gmm}_ali_${train_set}_sp_comb
tree_dir=exp/chain${nnet3_affix}/tree_bi${tree_affix}
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_comb_lats
#dir=exp/chain${nnet3_affix}/tdnn${tdnn_blstm_affix}_sp_bi
dir=exp/chain${nnet3_affix}/tdnn_blstm${tdnn_blstm_affix}_sp_bi_epoch${num_epochs}
if [ $label_delay -gt 0 ]; then dir=${dir}_ld${label_delay}; fi

train_data_dir=data/${train_set}_sp_hires_comb
lores_train_data_dir=data/${train_set}_sp_comb
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb


for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 14 ]; then
  echo "$0: creating lang directory with one state per phone."
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d data/lang_chain ]; then
    if [ data/lang_chain/L.fst -nt data/lang/L.fst ]; then
      echo "$0: data/lang_chain already exists, not overwriting it; continuing"
    else
      echo "$0: data/lang_chain already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang_s data/lang_chain
    silphonelist=$(cat data/lang_chain/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat data/lang_chain/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >data/lang_chain/topo
  fi
fi

if [ $stage -le 15 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --stage $align_fmllr_lats_stage --nj $nj --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang_s $gmm_dir $lat_dir
  #rm $lat_dir/fsts.*.gz # save space
fi
echo "This is debug point"
if [ $stage -le 16 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.
  if [ -f $tree_dir/final.mdl ]; then
    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
    exit 1;
  fi
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --leftmost-questions-truncate -1 \
      --cmd "$train_cmd" 4000 ${lores_train_data_dir} data/lang_chain $ali_dir $tree_dir
fi

if [ $stage -le 17 ]; then
  mkdir -p $dir

  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) delay=$label_delay affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=1024
  relu-renorm-layer name=tdnn2 input=Append(-1,0,1) dim=1024
  relu-renorm-layer name=tdnn3 input=Append(-1,0,1) dim=1024

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  lstmp-layer name=blstm1-forward cell-dim=1024 input=tdnn3 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 dropout-proportion=0.0 dropout-per-frame=true
  lstmp-layer name=blstm1-backward cell-dim=1024 input=tdnn3 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=3 dropout-proportion=0.0 dropout-per-frame=true
  #relu-renorm-layer name=tdnn4 input=Append(-3,0,3) dim=1024
  #relu-renorm-layer name=tdnn5 input=Append(-3,0,3) dim=1024
  lstmp-layer name=blstm2-forward cell-dim=1024 input=Append(blstm1-forward,blstm1-backward)  recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 dropout-proportion=0.0 dropout-per-frame=true
  lstmp-layer name=blstm2-backward cell-dim=1024 input=Append(blstm1-forward,blstm1-backward)  recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=3 dropout-proportion=0.0 dropout-per-frame=true
  #relu-renorm-layer name=tdnn6 input=Append(-3,0,3) dim=1024
  #relu-renorm-layer name=tdnn7 input=Append(-3,0,3) dim=1024
  lstmp-layer name=blstm3-forward cell-dim=1024 input=Append(blstm2-forward,blstm2-backward) recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 dropout-proportion=0.0 dropout-per-frame=true
  lstmp-layer name=blstm3-backward cell-dim=1024 input=Append(blstm2-forward,blstm2-backward) recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=3 dropout-proportion=0.0 dropout-per-frame=true

  ## adding the layers for chain branch
  output-layer name=output input=Append(blstm3-forward,blstm3-backward) output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=Append(blstm3-forward,blstm3-backward) output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/

fi

 

if [ $stage -le 18 ]; then

 steps/nnet3/chain/train.py --stage $train_stage --exit-stage $train_exit_stage  \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize 0.1 \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $chunk_width \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.num-chunk-per-minibatch $num_chunks_per_minibatch \
    --trainer.frames-per-iter $frames_per_iter \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.optimization.momentum 0.0 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.deriv-truncate-margin 8 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --use-gpu "wait" \
    --dir $dir
fi


if [ $stage -le 19 ] && [ $stop_stage -gt 19 ]  ; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_s_test_tgpr $dir $dir/graph
fi

if [ $stage -le 20 ] && [ $stop_stage -gt 20  ]  ; then
  for x in dev_s dev_t_16khz; do
    nspk=$(wc -l <data/$x/spk2utt)
    [ "$nspk" -gt "$decode_nj" ] && nspk=$decode_nj
    steps/nnet3/decode.sh --num-threads $num_threads_decode --nj $nspk --cmd "$decode_cmd" \
      --acwt 1.0 --post-decode-acwt 10.0 \
      --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${x}_hires \
      --scoring-opts "--min-lmwt 5 " \
      --extra-left-context $extra_left_context \
      --extra-right-context $extra_right_context \
      --frames-per-chunk $frames_per_chunk \
      $dir/graph data/${x}_hires $dir/decode_${x} || exit 1;
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_s_test_{tgpr,fgconst} \
      data/${x}_hires ${dir}/decode_${x} ${dir}/decode_${x}_rescore || exit
  done
fi

if [ $stage -le 21 ] && [ $stop_stage -gt 21 ]  ; then
  # evaluate gender-specific dev_s and dev_t
  #for x in dev_s dev_t_16khz; do
  for x in ${base_eval}_${gender_eval}  ; do
    nspk=$(wc -l <data/${x}_hires/spk2utt)
    [ "$nspk" -gt "$decode_nj" ] && nspk=$decode_nj
    steps/nnet3/decode.sh --num-threads $num_threads_decode --nj $nspk --cmd "$decode_cmd" \
      --use-gpu $use_gpu_decode \
      --acwt 1.0 --post-decode-acwt 10.0 \
      --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${base_eval}_hires \
      --scoring-opts "--min-lmwt 5 " \
      --iter $decode_iter \
      --extra-left-context $extra_left_context \
      --extra-right-context $extra_right_context \
      --frames-per-chunk $frames_per_chunk \
      $dir/graph data/${x}_hires $dir/decode_${x}${decode_iter_suffix} || exit 1;
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_s_test_{tgpr,fgconst} \
      data/${x}_hires ${dir}/decode_${x}${decode_iter_suffix} ${dir}/decode_${x}${decode_iter_suffix}_rescore || exit
  done
fi


echo "finished chain model..."

#exit 0
