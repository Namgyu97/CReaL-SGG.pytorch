# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.data import get_dataset_statistics
from .utils_motifs import rel_vectors, obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info 

from .utils_motifs import to_onehot, encode_box_info
from maskrcnn_benchmark.modeling.make_layers import make_fc




@registry.ROI_RELATION_PREDICTOR.register("ComplexPredictor")
class CReaL(nn.Module):
    def __init__(self, config, in_channels):
        super(CReaL, self).__init__()
        
        self.cfg = config
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        self.mlp_dim = 2048
        self.embed_dim = 300
        self.dropout_rate = 0.2

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels
        
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, _ = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)
        
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
        
        self.sub_trans = MLP(self.embed_dim, self.mlp_dim, self.mlp_dim, 2)
        self.obj_trans = MLP(self.embed_dim, self.mlp_dim, self.mlp_dim, 2)
        
        self.sub_gate = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.obj_gate = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.sub_fc = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.obj_fc = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.sub_dropout = nn.Dropout(self.dropout_rate)
        self.obj_dropout = nn.Dropout(self.dropout_rate)

        self.vis_to_sem = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), 
            nn.ReLU(True),
            nn.Dropout(self.dropout_rate), 
            nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        )
        self.pool_transform = MLP(self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM,
                                  self.mlp_dim, self.mlp_dim, 2)
        
        self.pred_transform = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        
        self.amp_real_mlp = nn.Linear(self.mlp_dim // 2, self.mlp_dim)
        self.amp_imag_mlp = nn.Linear(self.mlp_dim // 2, self.mlp_dim)
        self.amp_abs_mlp = nn.Linear(self.mlp_dim // 2, self.mlp_dim)
        self.rel_amp_fc = nn.Linear(self.mlp_dim * 3, self.mlp_dim)
        self.pred_amp_fc = nn.Linear(self.mlp_dim * 3, self.mlp_dim)
        self.pred_abs = MLP(self.mlp_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.pred_abs_dropout = nn.Dropout(self.dropout_rate)
        self.rel_abs_dropout = nn.Dropout(self.dropout_rate)
        
        self.phase_emb = MLP(self.mlp_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.sin_emb = nn.Linear(self.mlp_dim // 2, self.mlp_dim)
        self.cos_emb = nn.Linear(self.mlp_dim // 2, self.mlp_dim)
        self.rel_phs_fc = nn.Linear(self.mlp_dim * 3, self.mlp_dim)
        self.pred_phs_fc = nn.Linear(self.mlp_dim * 3, self.mlp_dim)
        self.pred_phs = MLP(self.mlp_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.pred_phs_dropout = nn.Dropout(self.dropout_rate)
        self.rel_phs_dropout = nn.Dropout(self.dropout_rate)
        self.norm_rel_amp = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_phs = nn.LayerNorm(self.mlp_dim)
        self.norm_pred_amp = nn.LayerNorm(self.mlp_dim)
        self.norm_pred_phs = nn.LayerNorm(self.mlp_dim)
        
        self.logit_scale_amp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_phs = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.uncertainty_fc = nn.Linear(self.num_rel_cls * 2, self.num_rel_cls)
        
        self.pos_embed = nn.Sequential(
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True)
        )
        
        self.pair_feat_tr = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(self.dropout_rate), nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        )
        self.pair_emb_tr = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(self.dropout_rate), nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        )

        self.uni_feat_tr = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(self.dropout_rate), nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        )
        self.uni_emb_tr = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(self.dropout_rate), nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        )

        self.pred_feat_tr = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(self.dropout_rate), nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        )
        self.pred_emb_tr = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(self.dropout_rate), nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        )

        self.uncertainty_fc = nn.Linear(self.num_rel_cls * 2, self.num_rel_cls)
        
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes)
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)
        
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

    def norm_state(self, c_state):
        norm = torch.linalg.norm(c_state, dim=-1, keepdim=True)
        return c_state / norm.clamp(min=1e-8)
    
    def complex_mul(self, a, b):
        real = a.real * b.real - a.imag * b.imag
        imag = a.real * b.imag + a.imag * b.real
        return self.norm_state(torch.complex(real, imag))

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        num_objs = [len(p) for p in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        
        entity_dists, entity_preds = self.refine_object_labels(roi_features, proposals)
        
        entity_features = self.post_emb(roi_features).view(-1, 2, self.mlp_dim)
        sub_feats_all = entity_features[:, 0].contiguous()
        obj_feats_all = entity_features[:, 1].contiguous()
        
        entity_embeds = self.obj_embed(entity_preds).split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        sub_feats_spl = sub_feats_all.split(num_objs, dim=0)
        obj_feats_spl = obj_feats_all.split(num_objs, dim=0)
        
        rel_states_list = []
        for pair_idx, sub_f, obj_f, emb in zip(rel_pair_idxs, sub_feats_spl, obj_feats_spl, entity_embeds):
            sub_embed = self.sub_trans(emb.index_select(0, pair_idx[:, 0]))
            obj_embed = self.obj_trans(emb.index_select(0, pair_idx[:, 1]))
            
            sub_sem = self.vis_to_sem(sub_f.index_select(0, pair_idx[:, 0]))
            obj_sem = self.vis_to_sem(obj_f.index_select(0, pair_idx[:, 1]))
            
            sub_gate_val = torch.sigmoid(self.sub_gate(torch.cat((sub_embed, sub_sem), dim=-1)))
            obj_gate_val = torch.sigmoid(self.obj_gate(torch.cat((obj_embed, obj_sem), dim=-1)))
            
            fused_sub = sub_embed + sub_sem * sub_gate_val
            fused_obj = obj_embed + obj_sem * obj_gate_val
            
            fused_sub = self.norm_sub(self.sub_dropout(F.relu(self.sub_fc(fused_sub))) + fused_sub)
            fused_obj = self.norm_obj(self.obj_dropout(F.relu(self.obj_fc(fused_obj))) + fused_obj)
            
            pairwise_feat = fusion_func(fused_sub, fused_obj)
            c_state = self.norm_state(torch.complex(self.pair_feat_tr(pairwise_feat), self.pair_emb_tr(pairwise_feat)))
            rel_states_list.append(c_state)
        
        rel_states = torch.cat(rel_states_list, dim=0)

        # ----- Complex -------
        union_base = self.pool_transform(union_features)
        union_state = self.norm_state(torch.complex(self.uni_feat_tr(union_base), self.uni_emb_tr(union_base)))
        rel_states = self.complex_mul(rel_states, union_state)
        
        rel_real, rel_imag = torch.real(rel_states), torch.imag(rel_states)
        rel_abs, rel_angle  = torch.abs(rel_states), torch.angle(rel_states)
        rel_sin, rel_cos  = torch.sin(rel_angle), torch.cos(rel_angle)
        
        # ----- Complex: Amp branch -------
        amp_input = torch.cat([rel_real, rel_imag, rel_abs], dim=-1)
        amp_input = self.rel_amp_fc(amp_input)
        amp_input = self.norm_rel_amp(amp_input)
        amp_act = F.gelu(amp_input)
        amp_dropout = self.rel_abs_dropout(amp_act)
        rel_amp_val = self.pred_abs(amp_dropout)
        rel_amp_norm = F.normalize(rel_amp_val, p=2, dim=1, eps=1e-8)
        
        # ----- Complex: Phs branch -------
        phs_input = torch.cat([rel_amp_norm, rel_sin, rel_cos], dim=-1)
        phs_input = self.rel_phs_fc(phs_input)
        phs_input = self.norm_rel_phs(phs_input)
        phs_act = F.gelu(phs_input)
        phs_dropout = self.rel_phs_dropout(phs_act)
        rel_phs_val = self.pred_phs(phs_dropout)
        rel_phs_norm = F.normalize(rel_phs_val, p=2, dim=1, eps=1e-8)
        
        # ----- Prototype ------
        pred_embed = self.pred_transform(self.rel_embed.weight)
        pred_states = self.norm_state(torch.complex(self.pred_feat_tr(pred_embed), self.pred_emb_tr(pred_embed)))
        
        pred_real, pred_imag = torch.real(pred_states), torch.imag(pred_states)
        pred_abs, pred_angle  = torch.abs(pred_states), torch.angle(pred_states)
        pred_sin, pred_cos  = torch.sin(pred_angle), torch.cos(pred_angle)
        
        # ----- Prototype: Amp branch ------
        pred_amp_input = torch.cat([pred_real, pred_imag, pred_abs], dim=-1)
        pred_amp_input = self.pred_amp_fc(pred_amp_input)
        pred_amp_act = F.gelu(pred_amp_input)
        pred_amp_dropout = self.pred_abs_dropout(pred_amp_act)
        pred_amp_val = self.pred_abs(pred_amp_dropout)
        pred_amp_norm = F.normalize(pred_amp_val, p=2, dim=1, eps=1e-8)
        
        # ----- Prototype: Phs branch ------
        pred_phs_input = torch.cat([pred_amp_norm, pred_sin, pred_cos], dim=-1)
        pred_phs_input = self.pred_phs_fc(pred_phs_input)
        pred_phs_act = F.gelu(pred_phs_input)
        pred_phs_dropout = self.pred_phs_dropout(pred_phs_act)
        pred_phs_val = self.pred_phs(pred_phs_dropout)
        pred_phs_norm = F.normalize(pred_phs_val, p=2, dim=1, eps=1e-8)

        rel_amp_logits   = rel_amp_norm @ pred_amp_norm.t() * self.logit_scale_amp.exp()
        rel_phase_logits = rel_phs_norm @ pred_phs_norm.t() * self.logit_scale_phs.exp()

        gate = torch.sigmoid(
            self.aaa_fc(torch.cat([rel_amp_logits, rel_phase_logits], dim=-1))
        )
        rel_dists = gate * rel_amp_logits + (1 - gate) * rel_phase_logits
        
        add_losses = {}
        if self.training and rel_labels is not None:
            # --- Prototype Regularization (Cosine similarity) ---
            # Amplitude branch
            simil_mat_amp = pred_amp_norm @ pred_amp_norm.t()
            l21_amp = torch.norm(torch.norm(simil_mat_amp, p=2, dim=1), p=1) / (self.num_rel_cls * self.num_rel_cls)
            add_losses.update({"amp_l21_loss": l21_amp})
            # Phase branch
            simil_mat_phs = pred_phs_norm @ pred_phs_norm.t()
            l21_phs = torch.norm(torch.norm(simil_mat_phs, p=2, dim=1), p=1) / (self.num_rel_cls * self.num_rel_cls)
            add_losses.update({"phs_l21_loss": l21_phs})
            
            ### Prototype Regularization  ---- Euclidean distance
            gamma = 30.0
            predicate_proto_a = pred_phs_val.unsqueeze(dim=1).expand(-1, 51, -1) 
            predicate_proto_b = pred_phs_val.detach().unsqueeze(dim=0).expand(51, -1, -1)
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1
            dist_loss = torch.max(torch.zeros(51).cuda(), -topK_proto_dis + gamma).mean()
            add_losses.update({"phs_dist_loss": dist_loss})

            # --- Prototype Clustering for amplitude branch ---
            margin = 0.10
            triplet_pred_pairs = []
            for label in unique_labels:
                sim_row = simil_mat_amp[label].clone()
                sim_row[label] = -1.0  # 자기 자신 제외
                most_similar_label = torch.argmax(sim_row)
                positive_mask = (rel_labels_cat == label)
                negative_mask = (rel_labels_cat == most_similar_label)
                if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                    positive_sample = rel_amp_norm[positive_mask].mean(dim=0)
                    negative_sample = rel_amp_norm[negative_mask].mean(dim=0)
                    triplet_pred_pairs.append((pred_amp_norm[label], positive_sample, negative_sample))
            
            pred_triplet_losses = []
            for anchor, positive, negative in triplet_pred_pairs:
                loss = nn.TripletMarginLoss(margin=margin, p=2)(anchor, positive, negative)
                pred_triplet_losses.append(loss)
            pred_triplet_loss = torch.stack(pred_triplet_losses).mean() if pred_triplet_losses else torch.tensor(0.0, device=roi_features.device)
            add_losses.update({"pred_triplet_loss": pred_triplet_loss})

            # --- Prototype Clustering for phase branch---
            margin = 0.10
            triplet_rel_pairs = []
            
            new_rel_pair_idxs = [entity_pred[pair_idx] for pair_idx, entity_pred in zip(rel_pair_idxs, entity_preds)]
            rel_pairs = torch.cat(new_rel_pair_idxs)
            rel_labels_cat = torch.cat(rel_labels)
            
            unique_labels = torch.unique(rel_labels_cat)
            unique_labels = unique_labels[unique_labels != 0]
            for label in unique_labels:
                same_rel_pairs = rel_pairs[rel_labels_cat == label]
                anchor_reps = rel_phs_norm[rel_labels_cat == label]
                for rel_pair, anchor_rep in zip(same_rel_pairs, anchor_reps):
                    same_pair_mask = (rel_pairs[:, 0] == rel_pair[0]) & (rel_pairs[:, 1] == rel_pair[1])
                    positive_mask = same_pair_mask & (rel_labels_cat == label)
                    negative_mask = same_pair_mask & (rel_labels_cat != label)
                    if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                        positive_sample = rel_phs_norm[positive_mask].mean(dim=0)
                        negative_sample = rel_phs_norm[negative_mask].mean(dim=0)
                        triplet_rel_pairs.append((anchor_rep, positive_sample, negative_sample))
            
            rel_triplet_losses = []
            for anchor, positive, negative in triplet_rel_pairs:
                loss = nn.TripletMarginLoss(margin=margin, p=2)(anchor, positive, negative)
                rel_triplet_losses.append(loss)
            rel_triplet_loss = torch.stack(rel_triplet_losses).mean() if rel_triplet_losses else torch.tensor(0.0, device=roi_features.device)
            add_losses.update({"triplet_loss": rel_triplet_loss})

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        return entity_dists, rel_dists, add_losses
        
    def refine_object_labels(self, roi_features, proposals):
        if self.mode == 'predcls':
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0).long()
            obj_embed = self.obj_embed(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight

        assert proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()
        
        return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists_split = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for idx in range(len(num_objs)):
            # is_overlap: (#box, #box, #class)
            is_overlap = nms_overlaps(boxes_per_cls[idx]).cpu().numpy() >= self.nms_thresh
            out_dists_sampled = F.softmax(obj_dists_split[idx], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1  # 배경 클래스 제외
            out_label = obj_dists_split[idx].new(num_objs[idx]).fill_(0)

            for j in range(num_objs[idx]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0
            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds

    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)  
        return x
    
    
def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2




def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
