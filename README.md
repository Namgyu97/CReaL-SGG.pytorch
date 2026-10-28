# Complex-Valued Relational Embeddings with Amplitude–Phase Decomposition for Scene Graph Generation


## Overview
![main](https://github.com/user-attachments/assets/20143e7d-b51d-4b4e-8ede-4659fb9c09f6)

## Abstract
Effectively modeling the subtle interplay of relational cues within compressed relational representations remains a challenge in Scene Graph Generation(SGG). Existing methods predominantly rely on real-valued embeddings that compress all related information into a single vector, thereby introducing inherent limitations that obscure the fine-grained distinctions essential for robust relational reasoning. To overcome these limitations, we propose CReaL, a novel complex-valued embedding framework that preserves the core information provided by real-valued embeddings while explicitly decomposing it into complementary amplitude and phase components. In our model, the amplitude branch quantifies the strength and prominence of visual features, whereas the phase branch encodes directional information by capturing geometric alignment and angular relationships between objects. In addition, We adopt two distinct training strategies: i) the amplitude branch is optimized to suppress inter-class similarity and reinforce visual strength, and ii) the phase branch is tuning to capture subtle differences among relation labels. Consequently, our approach achieves more precise and reliable inference of inter-object relationships by balancing the complementary contributions of the two branches. Experiments on the VG-150 and OI-V6 datasets show that CReaL outperforms state-of-the-art methods, particularly achieving up to a 6.7\% improvement in the PredCls task. Furthermore, visual analysis of the embedding space reveals that the amplitude and phase components naturally cluster into semantically meaningful groups, highlighting both the robustness and interpretability of our approach.

---

## Installation
Check [INSTALL.md](./INSTALL.md) for installation instructions.

## Dataset
Check [DATASET.md](./DATASET.md) for instructions of dataset preprocessing.

## Train
```
python3 \
  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR ComplexPredictor \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 \
  SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 1024 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 30000 \
  SOLVER.CHECKPOINT_PERIOD 30000 GLOVE_DIR ./datasets/vg/ \
  MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0;
```

## Device
All our experiments are conducted on one NVIDIA GeForce RTX 4090, if you wanna run it on your own device, make sure to follow distributed training instructions in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Acknowledgement
The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Citation
If you use this work, please cite:
```
@article{jung2026creal,
  author  = {Jung, NamGyu and Choi, Chang},
  title   = {Complex-Valued Relational Embeddings with Amplitude--Phase Decomposition for Scene Graph Generation},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2026},
  doi     = {10.1109/TNNLS.2026.3743378}
}
```
