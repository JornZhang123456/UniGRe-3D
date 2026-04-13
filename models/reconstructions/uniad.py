import copy
import random
from typing import Optional
from sklearn.neighbors import KDTree
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from models.initializer import initialize_from_cfg
from torch import Tensor, nn

class MultiScaleFeatureFusion(nn.Module):
   def __init__(self, channels, feature_size):
       super(MultiScaleFeatureFusion, self).__init__()
       self.feature_size = feature_size
       self.downsample_2x = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels))
       self.downsample_4x = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels))
       self.upsample_2x = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels))
       self.upsample_4x = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels))
       self.scale_attention = nn.MultiheadAttention(channels, num_heads=8, dropout=0.1)
       self.fusion_conv = nn.Sequential(nn.Linear(channels * 3, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels))
       self.scale_weights = nn.Parameter(torch.ones(3) / 3)

   def downsample_features(self, x, scale):
       G, B, C = x.shape
       if scale == 2:
           downsampled = x.view(G//2, 2, B, C).mean(dim=1)
           downsampled = self.downsample_2x(downsampled)
       elif scale == 4:
           downsampled = x.view(G//4, 4, B, C).mean(dim=1)
           downsampled = self.downsample_4x(downsampled)
       return downsampled

   def upsample_features(self, x, target_size, scale):
       G, B, C = x.shape
       if scale == 2:
           upsampled = x.repeat_interleave(2, dim=0)[:target_size]
           upsampled = self.upsample_2x(upsampled)
       elif scale == 4:
           upsampled = x.repeat_interleave(4, dim=0)[:target_size]
           upsampled = self.upsample_4x(upsampled)
       return upsampled

   def cross_scale_attention(self, features_list):
       all_features = torch.cat(features_list, dim=0)
       attended_features, _ = self.scale_attention(all_features, all_features, all_features)
       G = features_list[0].shape[0]
       scale1_out = attended_features[:G]
       scale2_out = attended_features[G:G+G//2]
       scale3_out = attended_features[G+G//2:]
       return [scale1_out, scale2_out, scale3_out]

   def forward(self, x):
       G, B, C = x.shape
       scale1_features = x
       scale2_features = self.downsample_features(x, scale=2)
       scale3_features = self.downsample_features(x, scale=4)
       attended_scales = self.cross_scale_attention([scale1_features, scale2_features, scale3_features])
       scale1_final = attended_scales[0]
       scale2_final = self.upsample_features(attended_scales[1], G, scale=2)
       scale3_final = self.upsample_features(attended_scales[2], G, scale=4)
       concatenated = torch.cat([scale1_final, scale2_final, scale3_final], dim=-1)
       fused_features = self.fusion_conv(concatenated)
       weights = F.softmax(self.scale_weights, dim=0)
       weighted_fusion = (weights[0] * scale1_final + weights[1] * scale2_final + weights[2] * scale3_final)
       final_output = fused_features + weighted_fusion
       return final_output

class LGFA(nn.Module):
    def __init__(self, channels, h_kernel_size=11, v_kernel_size=11, reduction=8):
        super(LGFA, self).__init__()
        hidden_dim = max(channels // reduction, 4)

        # 淇濈暀澶氬昂搴﹁瀺鍚?        self.multi_scale_fusion = MultiScaleFeatureFusion(channels, feature_size=64)

        # 鍒嗘敮涓€锛氭敼鎴愮湡姝ｆ部 token 缁村缓妯★紝鑰屼笉鏄闀垮害=1鍋氬嵎绉?        # 杈撳叆浼氫粠 [G, B, C] -> [B, C, G]
        self.caa_conv1 = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.h_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=h_kernel_size,
            stride=1,
            padding=h_kernel_size // 2,
            groups=hidden_dim
        )
        self.v_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=v_kernel_size,
            stride=1,
            padding=v_kernel_size // 2,
            groups=hidden_dim
        )
        self.caa_conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # 鍒嗘敮浜岋細淇濈暀浣犵殑 channel attention + global gate 鎬濊矾
        self.fa_ca = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels),
            nn.Sigmoid()
        )

        # 鍘熸潵杈撳嚭1锛岃繖閲屼繚鎸佲€滄敼鍔ㄦ渶灏忊€濓紝浠嶈緭鍑?
        self.fa_ga = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 鐢?logit 鍙傛暟 + sigmoid 绾︽潫铻嶅悎鏉冮噸鍒?[0,1]
        self.fusion_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
        x: [G, B, C]
        """
        x = self.multi_scale_fusion(x)
        G, B, C = x.shape

        # ========================
        # 鍒嗘敮涓€锛歵oken缁村眬閮ㄤ笂涓嬫枃 + 閫氶亾娉ㄦ剰鍔?        # ========================
        # [G, B, C] -> [B, C, G]
        x_token = x.permute(1, 2, 0).contiguous()

        conv1_out = self.caa_conv1(x_token)          # [B, C//r, G]
        h_context = self.h_conv(conv1_out)           # [B, C//r, G]
        v_context = self.v_conv(h_context)           # [B, C//r, G]
        caa_attn_factor = self.caa_conv2(v_context)  # [B, C, G]

        caa_out = (x_token * caa_attn_factor).permute(2, 0, 1).contiguous()  # [G, B, C]

        # ========================
        # 鍒嗘敮浜岋細閫氶亾娉ㄦ剰鍔?+ token绾у叏灞€闂ㄦ帶
        # ========================
        ca_weight = torch.mean(x, dim=0, keepdim=True)   # [1, B, C]
        ca_weight = self.fa_ca(ca_weight)                # [1, B, C]
        x_ca = x * ca_weight                             # [G, B, C]

        ga_weight = self.fa_ga(x_ca)                     # [G, B, 1]
        fa_out = x_ca * ga_weight                        # [G, B, C]

        # ========================
        # 铻嶅悎
        # ========================
        alpha = torch.sigmoid(self.fusion_weight)
        fused = alpha * caa_out + (1.0 - alpha) * fa_out

        return fused


class HPRM(nn.Module):
    def __init__(self, hidden_dim, num_levels=3, expansion_factor=2):
        super(HPRM, self).__init__()
        mid_dim = hidden_dim * expansion_factor

        self.coarse_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        self.medium_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        self.fine_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, hidden_dim),
            nn.Dropout(0.1)
        )

        # 鍘熸潵鐨勯潤鎬?level_weights 鏀规垚鍔ㄦ€?token-wise level selector
        self.level_selector = nn.Sequential(
            nn.Linear(hidden_dim * num_levels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_levels)
        )

        # 璐ㄩ噺闂ㄦ帶锛氬姞鍏ュ師濮嬭緭鍏?x锛屼竴鍏?浠界壒寰?        self.quality_gate = nn.Sequential(
            nn.Linear(hidden_dim * (num_levels + 1), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [..., hidden_dim]
        杈撳嚭 shape 涓?x 鐩稿悓
        """
        coarse_out = self.coarse_reconstructor(x)
        medium_out = self.medium_reconstructor(coarse_out + x)
        fine_out = self.fine_reconstructor(medium_out + x)

        # 鎷兼帴涓夊眰杈撳嚭锛屽姩鎬侀娴嬫瘡涓?token 鐨?level 鏉冮噸
        concatenated = torch.cat([coarse_out, medium_out, fine_out], dim=-1)   # [..., 3C]
        level_logits = self.level_selector(concatenated)                       # [..., 3]
        level_weights = F.softmax(level_logits, dim=-1)                        # [..., 3]

        weighted_output = (
            coarse_out * level_weights[..., 0:1] +
            medium_out * level_weights[..., 1:2] +
            fine_out * level_weights[..., 2:3]
        )

        # gate 閲屽姞鍏ュ師濮嬭緭鍏?        gate_input = torch.cat([x, coarse_out, medium_out, fine_out], dim=-1)  # [..., 4C]
        quality_weights = self.quality_gate(gate_input)                         # [..., C]

        # 姣斿師鏉ョ殑 weighted_output * gate + x * (1-gate) 鏇寸洿瑙?        final_output = x + quality_weights * (weighted_output - x)

        return final_output


def flip_normals_to_outward(points, normals):
    """
    Flip the normal direction to ensure it faces outwards.

    Args:
        points (np.ndarray): Point cloud coordinates, shape is (N, 3)
        normals (np.ndarray): Point cloud normal, shape (N, 3)

    Returns:
        np.ndarray: Normals of uniform orientation, of shape (N, 3)
    """
    centroid = np.mean(points, axis=0)
    directions = points - centroid
    dot_products = np.sum(normals * directions, axis=1)
    normals[dot_products < 0] = -normals[dot_products < 0]
    return normals
def estimate_parameters_kdtree(points, k=7, sample_size=1000):
    num_points = points.shape[0]
    sample_size = min(sample_size, num_points)
    kdtree = KDTree(points)
    sampled_indices = np.random.choice(num_points, sample_size, replace=False)
    sampled_points = points[sampled_indices]

    distances, _ = kdtree.query(sampled_points, k=2)
    avg_distance = np.mean(distances[:, 1])

    radius = k * avg_distance
    max_nn = min(50, max(20, int(num_points / 1000)))

    return radius, max_nn
def analyze_normals_curvatures_multiscale(
    point_cloud,
    k=7,
    radius=None,
    radius_multipliers=(0.5, 1.0, 1.5),
    radius_weights=None,
):
    """
    Multi-scale normal and curvature variation analysis for batched point clouds.
    """
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()

    B, N, _ = point_cloud.shape
    fused_normal_variations = np.zeros((B, N))
    fused_curvature_variations = np.zeros((B, N))

    for b in range(B):
        group = point_cloud[b]
        kdtree = KDTree(group)

        base_radius = radius
        if base_radius is None:
            base_radius, _ = estimate_parameters_kdtree(group, k=k)
        radii = [base_radius * m for m in radius_multipliers]

        # 濡傛灉鏈寚瀹氭潈閲嶏紝鍒欏 multiplier==1 鐨勫昂搴﹀姞鏉冩洿楂?        if radius_weights is None:
            weights = []
            for m in radius_multipliers:
                w = 2.0 if np.isclose(m, 1.0) else 1.0
                weights.append(w)
            weights = np.array(weights, dtype=np.float64)
        else:
            weights = np.array(radius_weights, dtype=np.float64)
        weights = weights / (weights.sum() + 1e-8)

        normal_vars_scales = []
        curvature_vars_scales = []
        for r in radii:
            normals = np.zeros((N, 3))
            curvatures = np.zeros((N))

            for i in range   (N):
                idx = kdtree.query_radius(group[i].reshape(1, -1), r=r)[0]
                neighbors = group[idx]

                if len(neighbors) < 3:
                    continue

                centroid = np.mean(neighbors, axis=0)
                cov_matrix = np.cov((neighbors - centroid).T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                normals[i] = eigenvectors[:, 0]
                curvatures[i] = eigenvalues[0] / (np.sum(eigenvalues) + 1e-8)

            normals = flip_normals_to_outward(group, normals)

            normal_variations = np.zeros((N))
            curvature_variations = np.zeros((N))
            for i in range(N):
                idx = kdtree.query_radius(group[i].reshape(1, -1), r=r)[0]
                if len(idx) == 0:
                    continue
                neighbor_normals = normals[idx]

                dot_products = np.dot(neighbor_normals, normals[i])
                dot_products = np.clip(dot_products, -1.0, 1.0)
                angles = np.arccos(dot_products)
                normal_variations[i] = np.mean(angles)

                neighbor_curvatures = curvatures[idx]
                curvature_variations[i] = np.mean(
                    np.abs(curvatures[i] - neighbor_curvatures)
                )

            normal_vars_scales.append(normal_variations)
            curvature_vars_scales.append(curvature_variations)

        normal_stack = np.stack(normal_vars_scales, axis=0)
        curvature_stack = np.stack(curvature_vars_scales, axis=0)
        fused_normal_variations[b] = np.sum(normal_stack * weights[:, None], axis=0)
        fused_curvature_variations[b] = np.sum(curvature_stack * weights[:, None], axis=0)

    return fused_normal_variations, fused_curvature_variations
class UniAD(nn.Module):
   def __init__(self, feature_size, feature_jitter, neighbor_mask, hidden_dim, initializer, cls_num, inplanes=1152, k=5, mask_ratio=0.4, radius_multipliers=(0.5, 1.0, 1.5),radius_weights=None,**kwargs):
       super().__init__()
       self.feature_jitter = feature_jitter
       self.cls_num = cls_num
       self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, hidden_dim))
       self.distence = torch.nn.MSELoss()
       self.transformer = Transformer(hidden_dim, feature_size, neighbor_mask, **kwargs)
       self.input_proj = nn.Linear(inplanes, hidden_dim)
       self.output_proj = nn.Linear(hidden_dim, inplanes)
       self.cls_head_finetune = nn.Sequential(nn.Linear(inplanes*2, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, self.cls_num))
       self.gem_dict = {}
       self.k = k
       self.mask_ratio = mask_ratio
       self.radius_multipliers = radius_multipliers
       self.radius_weights = radius_weights

       initialize_from_cfg(self, initializer)

   def add_jitter(self, feature_tokens, scale, prob):
       if random.uniform(0, 1) <= prob:
           num_tokens, batch_size, dim_channel = feature_tokens.shape
           feature_norms = (feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel)
           jitter = torch.randn((num_tokens, batch_size, dim_channel)).cuda()
           jitter = jitter * feature_norms * scale
           feature_tokens = feature_tokens + jitter
       return feature_tokens

   def forward(self, input):
       feature_align = input["xyz_features"]
       center = input["center"]
       filename = input["filename"]
       filename = filename[0]
       if filename not in self.gem_dict:
           normal_variations, curvature_variations = analyze_normals_curvatures_multiscale(
               center,
               k=self.k,
               radius_multipliers=self.radius_multipliers,
               radius_weights=self.radius_weights,
           )
           geome_vars = normal_variations + 10 * curvature_variations
           self.gem_dict[filename] = geome_vars
       geome_vars = self.gem_dict[filename]
       if isinstance(geome_vars, np.ndarray):
           geome_vars = torch.from_numpy(geome_vars).float().to(center.device)

       #geome_vars = torch.zeros(center.shape[0], center.shape[1]).cuda()
       feature_tokens = rearrange(feature_align, "b n g -> g b n")
       if self.training and self.feature_jitter:
           feature_tokens = self.add_jitter(feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob)
       feature_tokens = self.input_proj(feature_tokens)
       pos_embed = self.pos_embed(center).permute(1,0,2)
       output_decoder, _ = self.transformer(feature_tokens, pos_embed ,geome_vars,self.mask_ratio)
       feature_rec_tokens = self.output_proj(output_decoder)
       feature_rec = rearrange(feature_rec_tokens, "g b n -> b n g")
       feature_cls = feature_rec.detach().clone()
       feature_cls.requires_grad = True
       feature_cls = rearrange(feature_cls,"b n g -> b g n")
       concat_f = torch.cat([feature_cls[:, 0], feature_cls[:, 1:].max(1)[0]], dim=-1)
       cls_pred = self.cls_head_finetune(concat_f)
       pred = torch.sqrt(torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True))
       return {"feature_rec": feature_rec, "feature_align": feature_align, "pred": pred, "cls_pred": cls_pred}

class Transformer(nn.Module):
   def __init__(self, hidden_dim, feature_size, neighbor_mask, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1, activation="relu", normalize_before=False, return_intermediate_dec=False):
       super().__init__()
       self.feature_size = feature_size
       encoder_layer = TransformerEncoderLayerWithLGFA(hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before)
       encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
       self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
       decoder_layer = TransformerDecoderLayerWithHPRM(hidden_dim, feature_size, nhead, dim_feedforward, dropout, activation, normalize_before)
       decoder_norm = nn.LayerNorm(hidden_dim)
       self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
       self.hidden_dim = hidden_dim
       self.nhead = nhead

   def forward(self, src, pos_embed, geome_vars,mask_ratio):
       output_encoder = self.encoder(src, src_key_padding_mask=None, pos=pos_embed)
       output_decoder = self.decoder(output_encoder, tgt_key_padding_mask=None, pos=pos_embed)
       return output_decoder, output_encoder

class TransformerEncoder(nn.Module):
   def __init__(self, encoder_layer, num_layers, norm=None):
       super().__init__()
       self.layers = _get_clones(encoder_layer, num_layers)
       self.num_layers = num_layers
       self.norm = norm

   def forward(self, src, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
       output = src
       for layer in self.layers:
           output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
       if self.norm is not None:
           output = self.norm(output)
       return output

class TransformerDecoder(nn.Module):
   def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
       super().__init__()
       self.layers = _get_clones(decoder_layer, num_layers)
       self.num_layers = num_layers
       self.norm = norm
       self.return_intermediate = return_intermediate

   def forward(self, memory, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
       output = memory
       intermediate = []
       for layer in self.layers:
           output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos)
           if self.return_intermediate:
               intermediate.append(self.norm(output))
       if self.norm is not None:
           output = self.norm(output)
           if self.return_intermediate:
               intermediate.pop()
               intermediate.append(output)
       if self.return_intermediate:
           return torch.stack(intermediate)
       return output

class TransformerEncoderLayerWithLGFA(nn.Module):
   def __init__(self, hidden_dim, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
       super().__init__()
       self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
       self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
       self.dropout = nn.Dropout(dropout)
       self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
       self.norm1 = nn.LayerNorm(hidden_dim)
       self.norm2 = nn.LayerNorm(hidden_dim)
       self.norm3 = nn.LayerNorm(hidden_dim)
       self.dropout1 = nn.Dropout(dropout)
       self.dropout2 = nn.Dropout(dropout)
       self.dropout3 = nn.Dropout(dropout)
       self.lgfa = LGFA(hidden_dim)
       self.activation = _get_activation_fn(activation)
       self.normalize_before = normalize_before

   def with_pos_embed(self, tensor, pos):
       return tensor if pos is None else tensor + pos

   def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
       q = k = self.with_pos_embed(src, pos)
       src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
       src = src + self.dropout1(src2)
       src = self.norm1(src)
       src2 = self.lgfa(src)
       src = src + self.dropout3(src2)
       src = self.norm3(src)
       src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
       src = src + self.dropout2(src2)
       src = self.norm2(src)
       return src

   def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
       src2 = self.norm1(src)
       q = k = self.with_pos_embed(src2, pos)
       src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
       src = src + self.dropout1(src2)
       src2 = self.norm3(src)
       src2 = self.lgfa(src2)
       src = src + self.dropout3(src2)
       src2 = self.norm2(src)
       src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
       src = src + self.dropout2(src2)
       return src

   def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
       if self.normalize_before:
           return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
       return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayerWithHPRM(nn.Module):
   def __init__(self, hidden_dim, feature_size, nhead, dim_feedforward, dropout=0.1, activation="relu", normalize_before=False):
       super().__init__()
       num_queries = feature_size
       self.learned_embed = nn.Embedding(num_queries, hidden_dim)
       self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
       self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
       self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
       self.dropout = nn.Dropout(dropout)
       self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
       self.norm1 = nn.LayerNorm(hidden_dim)
       self.norm2 = nn.LayerNorm(hidden_dim)
       self.norm3 = nn.LayerNorm(hidden_dim)
       self.norm4 = nn.LayerNorm(hidden_dim)
       self.dropout1 = nn.Dropout(dropout)
       self.dropout2 = nn.Dropout(dropout)
       self.dropout3 = nn.Dropout(dropout)
       self.dropout4 = nn.Dropout(dropout)
       self.hprm = HPRM(hidden_dim)
       self.activation = _get_activation_fn(activation)
       self.normalize_before = normalize_before

   def with_pos_embed(self, tensor, pos: Optional[Tensor]):
       return tensor if pos is None else tensor + pos

   def forward_post(self, out, memory, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
       tgt = pos
       tgt2 = self.self_attn(query=tgt, key=self.with_pos_embed(memory, pos), value=memory, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
       tgt = tgt + self.dropout1(tgt2)
       tgt = self.norm1(tgt)
       tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos), key=self.with_pos_embed(out, pos), value=out, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
       tgt = tgt + self.dropout2(tgt2)
       tgt = self.norm2(tgt)
       tgt2 = self.hprm(tgt)
       tgt = tgt + self.dropout4(tgt2)
       tgt = self.norm4(tgt)
       tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       tgt = tgt + self.dropout3(tgt2)
       tgt = self.norm3(tgt)
       return tgt

   def forward_pre(self, out, memory, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
       tgt = pos
       tgt2 = self.norm1(tgt)
       tgt2 = self.self_attn(query=self.with_pos_embed(tgt2, pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
       tgt = tgt + self.dropout1(tgt2)
       tgt2 = self.norm2(tgt)
       tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, pos), key=self.with_pos_embed(out, pos), value=out, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
       tgt = tgt + self.dropout2(tgt2)
       tgt2 = self.norm4(tgt)
       tgt2 = self.hprm(tgt2)
       tgt = tgt + self.dropout4(tgt2)
       tgt2 = self.norm3(tgt)
       tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
       tgt = tgt + self.dropout3(tgt2)
       return tgt

   def forward(self, out, memory, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
       if self.normalize_before:
           return self.forward_pre(out, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos)
       return self.forward_post(out, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos)

def _get_clones(module, N):
   return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
   if activation == "relu":
       return F.relu
   if activation == "gelu":
       return F.gelu
   if activation == "glu":
       return F.glu
   raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
