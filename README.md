# Complex-Valued Relational Embeddings with Amplitude-Phase Decomposition for Scene Graph Generation


## Overview
![main](https://github.com/user-attachments/assets/20143e7d-b51d-4b4e-8ede-4659fb9c09f6)

## Abstract
Learning compact and discriminative representations of object relations is central to structured visual reasoning. In scene graph generation, these relations must be inferred from ambiguous visual cues under a heavy class imbalance; yet most methods encode a relation as a single real-valued vector that entangles interaction strength with a discriminative orientation, so frequent predicates overwrite the distinctions among rare ones. We propose CReaL, which fuses the subject–object pair with its context through complex multiplication, letting the context scale the magnitude and shift the phase of the relation. The result is split into an amplitude branch trained for interaction strength and a phase branch for the angular structure that discriminates similar relations, with branch-specific objectives that keep the two disentangled. A dynamic gating mechanism then fuses them with per-class weights computed from both branches, and the framework stays compatible with standard real-valued pipelines. On Visual Genome 150 (VG-150), CReaL improves mR@100 across predicate classification (PredCls), scene graph classification (SGCls), and scene graph detection (SGDet) over state-of-the-art methods, including a 5.9% gain in PredCls over the strongest knowledge-graph-based baseline despite using no external knowledge graph, and it attains the best overall score on Open Images v6 (OI-V6). Visualization and clustering analyses show that the amplitude and phase form complementary structures whose gated combination sharpens decision boundaries, indicating that explicit relational decomposition is an effective foundation for fine-grained relational understanding.

---

## Installation
Check [INSTALL.md](./INSTALL.md) for installation instructions.

## Dataset
Check [DATASET.md](./DATASET.md) for instructions on dataset preprocessing.

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
All our experiments are conducted on one NVIDIA GeForce RTX 4090. If you want to run it on your own device, make sure to follow distributed training instructions in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Acknowledgement
The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Citation
If you use this work, please cite:
```
@article{jung2026creal,
  author  = {Jung, NamGyu and Choi, Chang},
  title   = {Complex-Valued Relational Embeddings with Amplitude-Phase Decomposition for Scene Graph Generation},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2026},
  doi     = {10.1109/TNNLS.2026.3743378}
}
```
