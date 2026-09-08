from torch import nn
import torch
import torch.nn.functional as F
from models.transformer import Transformer
import math
from torch.autograd import Variable


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=136):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = Variable(self.pe[:x.size(0), :], requires_grad=False)

        x = x + pe
        return self.dropout(x)


def generate_modality_mask():
    """Generate a mask combination that includes at least one modality"""
    combinations = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1]
    ]
    idx = torch.randint(0, 6, [1])
    return combinations[idx[0]]


class LGP_Gate(nn.Module):
    """
    Multi-QuAD LGP (Eq.2) 门控：无额外质量假设，仅靠跨模态上下文重加权每维。

    流程：
    1. 池化三路 encoder 特征，concat 三路 δ_v（NFCE 精神：cos 相似度）
    2. MLP 生成每路门控参数 W,b
    3. q = σ(I·W + b)，I' = q ⊙ I（式(2)，无残差，不破坏原始分布）
    4. Loss 里加 Δ 近似项：鼓励 gate 后的特征比 gate 前对任务更有区分力

    最小实现，不引入「质量高低」的假设，直接是「数据驱动学出的维重加权」。
    """

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proto_v = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        self.proto_a = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        self.proto_f = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)

        ctx_dim = 3 * hidden_dim + 3
        self.ctx_mlp = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pd_v = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.pd_a = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.pd_f = nn.Linear(hidden_dim, 2 * hidden_dim)

    @staticmethod
    def _delta_v(h, proto):
        h_n = F.normalize(h, dim=-1, eps=1e-8)
        p_n = F.normalize(proto, dim=-1, eps=1e-8)
        cos = torch.mm(h_n, p_n.t())
        return torch.softmax(cos, dim=-1).max(dim=-1, keepdim=True).values

    def forward(self, proj_v, proj_a, proj_f, presence):
        """
        proj_v/a/f: (B,T,D) encoder 输出原始特征
        presence: (B,3) float {0,1}
        返回：(gated_v, gated_a, gated_f) + dict(Δ项所需中间量)
        """
        device, dtype = proj_v.device, proj_v.dtype
        presence = presence.to(device=device, dtype=dtype)

        hv = proj_v.mean(dim=1)
        ha = proj_a.mean(dim=1)
        hf = proj_f.mean(dim=1)

        dv = self._delta_v(hv, self.proto_v)
        da = self._delta_v(ha, self.proto_a)
        df = self._delta_v(hf, self.proto_f)

        ctx = torch.cat([hv, ha, hf, presence], dim=-1)
        shared = self.ctx_mlp(ctx)

        def _gate(proj, pd_layer, dv_):
            w, b = pd_layer(shared).chunk(2, dim=-1)
            q = torch.sigmoid(proj * w.unsqueeze(1) + b.unsqueeze(1))
            alpha = 0.05
            gated = proj + alpha * (q - 0.5) * proj
            return gated, q

        gated_v, q_v = _gate(proj_v, self.pd_v, dv)
        gated_a, q_a = _gate(proj_a, self.pd_a, da)
        gated_f, q_f = _gate(proj_f, self.pd_f, df)

        gated_v = gated_v * presence[:, 0:1].unsqueeze(-1)
        gated_a = gated_a * presence[:, 1:2].unsqueeze(-1)
        gated_f = gated_f * presence[:, 2:3].unsqueeze(-1)

        return gated_v, gated_a, gated_f, dict(q_v=q_v, q_a=q_a, q_f=q_f,
                                               dv=dv, da=da, df=df,
                                               proj_v=proj_v, proj_a=proj_a, proj_f=proj_f)


class ExtremeModalityGenerator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Cross-modal attention enhancement
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True,
                kdim=hidden_dim,  # Explicitly specify key dimension
                vdim=hidden_dim  # Explicitly specify value dimension
            )
            for _ in range(2)  # Two layers of attention
        ])
        # Conditional gating
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, exist_feats, mask_feat):
        """exist_feats: List of features from existing modalities (e.g., pass [v,f] when generating audio)"""
        # Concatenate existing modality features
        context = torch.cat(exist_feats, dim=1)
        # Multi-level attention
        for attn in self.cross_attn:
            mask_feat, _ = attn(mask_feat, context, context)
        # Gated fusion
        gate = self.gate(torch.cat([mask_feat.mean(1), context.mean(1)], -1))
        return gate.unsqueeze(1) * mask_feat


class CrossModalBottleneck(nn.Module):
    """DCER Stage-2: Cross-Modal Bottleneck Tokens.

    Minimal implementation following Eq.(2):
        Z = softmax(Q H^T / sqrt(D)) H
    where H is concatenated modality features and Q are learnable bottleneck tokens.

    Inputs are already high-level features (e.g., VST/I3D/AST), so we only implement
    the cross-modal bottleneck without additional frequency-domain compression.
    """

    def __init__(self, hidden_dim: int, k_tokens: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_tokens = k_tokens
        self.q = nn.Parameter(torch.randn(k_tokens, hidden_dim) * 0.02)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = Mlp(hidden_dim, hidden_dim * 2, hidden_dim)

    def forward(self, feats_list):
        """feats_list: list[(B,T,D)] -> Z: (B,K,D)

        If feats_list is empty (should not happen), returns zeros.
        """
        if len(feats_list) == 0:
            return None
        h = torch.cat(feats_list, dim=1)  # (B,S,D)
        b, _, d = h.shape
        q = self.q.unsqueeze(0).expand(b, -1, -1)  # (B,K,D)
        scale = 1.0 / math.sqrt(d)
        attn = torch.softmax(torch.bmm(q, h.transpose(1, 2)) * scale, dim=-1)  # (B,K,S)
        z = torch.bmm(attn, h)  # (B,K,D)
        z = self.norm(z)
        z = z + self.ffn(z)
        return z


class EnergyBasedReconstructor(nn.Module):
    """DCER Stage-3: Energy-Based Reconstruction for missing modalities.

    We reconstruct missing modality features h_m by minimizing a learned energy
    E(h_m; Z, H_obs) via a small number of gradient steps.
    """

    def __init__(self, hidden_dim: int, lambda_reg: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        # Energy network outputs a scalar per sample.
        self.energy_net = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def energy(self, h_m, h_obs_pool, z):
        """Compute energy scalar per batch element.

        h_m: (B,T,D) candidate missing feature
        h_obs_pool: (B,D) pooled observed context
        z: (B,K,D) bottleneck tokens
        """
        b, _, d = h_m.shape
        scale = 1.0 / math.sqrt(d)
        # Cross-attend: h_m (query) to Z (key/value)
        attn = torch.softmax(torch.bmm(h_m, z.transpose(1, 2)) * scale, dim=-1)  # (B,T,K)
        z_ctx = torch.bmm(attn, z)  # (B,T,D)
        diff_pool = (h_m - z_ctx).mean(dim=1)  # (B,D)
        z_pool = z.mean(dim=1)  # (B,D)
        x = torch.cat([diff_pool, h_obs_pool, z_pool], dim=-1)  # (B,3D)
        e = F.softplus(self.energy_net(x)).squeeze(-1)  # (B,)
        # Simple regularization to prevent exploding h_m.
        e = e + self.lambda_reg * (h_m.mean(dim=1).pow(2).sum(dim=-1))
        return e

    def forward(self, exist_feats, z, n_steps: int = 3, step_size: float = 0.1, momentum: float = 0.0,
                unroll: bool = True):
        """Reconstruct missing modality feature.

        exist_feats: list[(B,T,D)] observed modalities
        z: (B,K,D) bottleneck tokens computed from observed modalities
        Returns: (h_rec, e_final)
            h_rec: (B,T,D)
            e_final: (B,) final energy
        """
        if len(exist_feats) == 0:
            raise ValueError("EnergyBasedReconstructor requires at least one observed modality.")

        # Initialize from observed modalities mean (more stable than zeros)
        h_obs = torch.stack(exist_feats, dim=0).mean(dim=0)  # (B,T,D)
        h = h_obs.clone()
        # Momentum buffer
        v = torch.zeros_like(h)

        # Pool observed context once
        h_obs_pool = h_obs.mean(dim=1)  # (B,D)

        for _ in range(int(n_steps)):
            h = h.requires_grad_(True)
            e = self.energy(h, h_obs_pool, z).mean()
            grad = torch.autograd.grad(e, h, create_graph=bool(unroll), retain_graph=bool(unroll))[0]
            if momentum and momentum > 0:
                v = momentum * v + grad
                step = v
            else:
                step = grad
            h = h - float(step_size) * step
            # If not unrolling, stop higher-order grads to keep memory stable.
            if not unroll:
                h = h.detach()

        e_final = self.energy(h, h_obs_pool, z)  # (B,)
        return h, e_final


class MoE_AQA(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder,
                 n_query, dropout, config):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        # Initialize projection network
        self.in_proj = nn.ModuleDict({
            'v': self._build_proj(in_dim, hidden_dim),
            'a': self._build_proj(768, hidden_dim),
            'f': self._build_proj(1024, hidden_dim)
        })

        # Expert system reconstruction
        self.experts = nn.ModuleDict({
            'v': self._build_expret(hidden_dim, hidden_dim),
            'a': self._build_expret(hidden_dim, hidden_dim),
            'f': self._build_expret(hidden_dim, hidden_dim),
        })

        # Missing modality generation
        self.generators = nn.ModuleDict({
            'v': ExtremeModalityGenerator(self.hidden_dim),
            'a': ExtremeModalityGenerator(self.hidden_dim),
            'f': ExtremeModalityGenerator(self.hidden_dim)
        })

        # Routing network enhancement
        self.router_mix = Mlp(hidden_dim, hidden_dim // 2, 3)

        # Decoding module
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )

        # Feature fusion module
        self.fusion = nn.Sequential(
            nn.Conv1d(3 * hidden_dim, 2 * hidden_dim, kernel_size=1),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU(True),
            nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim)
        )

        self.prototype = nn.Embedding(n_query, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, n_query)
        self.register_buffer('weight', torch.linspace(0, 1, n_query))
        self.pos_encoder = PositionalEncoding(hidden_dim)

        self.lgp_gate = LGP_Gate(hidden_dim, dropout=dropout)

        # DCER-style modules (Stage2+Stage3) for missing modality handling
        bottleneck_k = getattr(config, 'bottleneck_k', 4)
        self.bottleneck = CrossModalBottleneck(hidden_dim, k_tokens=bottleneck_k)
        ebr_lambda = getattr(config, 'ebr_lambda_reg', 0.1)
        # Use modality-specific energy functions E_m (minimal but closer to DCER formulation)
        self.ebr = nn.ModuleDict({
            'v': EnergyBasedReconstructor(hidden_dim, lambda_reg=ebr_lambda),
            'a': EnergyBasedReconstructor(hidden_dim, lambda_reg=ebr_lambda),
            'f': EnergyBasedReconstructor(hidden_dim, lambda_reg=ebr_lambda),
        })

    def _build_expret(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim // 2),
            nn.GELU(),
            nn.Conv1d(out_dim // 2, out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim)
        )

    def _build_proj(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, in_dim // 2, kernel_size=1),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(True),
            nn.Conv1d(in_dim // 2, out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, video, audio, flow, mask):
        if self.training:
            return self.forward_train(video, audio, flow, mask)
        else:
            return self.forward_test(video, audio, flow, mask)

    def forward_train(self, video, audio, flow, mask):
        b, t, c = video.shape
        
        # Feature projection
        v = self.in_proj['v'](video.transpose(1, 2)).transpose(1, 2)
        a = self.in_proj['a'](audio.transpose(1, 2)).transpose(1, 2)
        f = self.in_proj['f'](flow.transpose(1, 2)).transpose(1, 2)
        proj_v = self.transformer.encoder(v)
        proj_a = self.transformer.encoder(a)
        proj_f = self.transformer.encoder(f)

        # LGP Gate (Multi-QuAD Eq.2)
        # 1) full-context gated features (for reconstruction target + joint bottleneck)
        presence_full = torch.ones(b, 3, device=proj_v.device, dtype=proj_v.dtype)
        if getattr(self.config, 'no_lgp', False):
            gated_full_v, gated_full_a, gated_full_f = proj_v, proj_a, proj_f
            gate_info = None
        else:
            gated_full_v, gated_full_a, gated_full_f, gate_info = self.lgp_gate(proj_v, proj_a, proj_f, presence_full)
        use_decr = not getattr(self.config, 'no_decr', False)
        z_full = self.bottleneck([gated_full_v, gated_full_a, gated_full_f]) if use_decr else None

        # 2) simulate missing modalities (at least one modality present)
        v_pres, a_pres, f_pres = generate_modality_mask()
        presence = torch.tensor([[v_pres, a_pres, f_pres]], device=proj_v.device,
                                dtype=proj_v.dtype).expand(b, 3)

        proj_v_in = proj_v if v_pres == 1 else torch.zeros_like(proj_v)
        proj_a_in = proj_a if a_pres == 1 else torch.zeros_like(proj_a)
        proj_f_in = proj_f if f_pres == 1 else torch.zeros_like(proj_f)

        if getattr(self.config, 'no_lgp', False):
            gated_v, gated_a, gated_f = proj_v_in, proj_a_in, proj_f_in
        else:
            gated_v, gated_a, gated_f, _ = self.lgp_gate(proj_v_in, proj_a_in, proj_f_in, presence)

        v_mask = gated_v
        a_mask = gated_a
        f_mask = gated_f

        # 3) DCER Stage2+Stage3: bottleneck + energy-based reconstruction
        # Build observed list and reconstruct missing sequentially.
        recon_loss = torch.tensor(0.0, device=proj_v.device, dtype=proj_v.dtype)
        if use_decr:
            recon_mse = 0.0
            energy_term = 0.0

            exist_feats = []
            if v_pres == 1:
                exist_feats.append(v_mask)
            if a_pres == 1:
                exist_feats.append(a_mask)
            if f_pres == 1:
                exist_feats.append(f_mask)

            missing = []
            if v_pres == 0:
                missing.append('v')
            if a_pres == 0:
                missing.append('a')
            if f_pres == 0:
                missing.append('f')

            cfg = self.config
            ebr_steps = int(getattr(cfg, 'ebr_steps', 3))
            ebr_lr = float(getattr(cfg, 'ebr_lr', 0.1))
            ebr_momentum = float(getattr(cfg, 'ebr_momentum', 0.0))
            ebr_unroll = bool(getattr(cfg, 'ebr_unroll', True))
            beta_energy = float(getattr(cfg, 'ebr_beta_energy', 0.01))
            gamma_joint = float(getattr(cfg, 'ebr_gamma_joint', 0.05))

            for m in missing:
                # Recompute Z from currently available features (observed + already reconstructed)
                z_obs = self.bottleneck(exist_feats)
                h_rec, e_final = self.ebr[m](exist_feats, z_obs, n_steps=ebr_steps, step_size=ebr_lr,
                                                momentum=ebr_momentum, unroll=ebr_unroll)
                energy_term = energy_term + e_final.mean()
                if m == 'v':
                    v_mask = h_rec
                    recon_mse = recon_mse + F.mse_loss(v_mask, gated_full_v)
                elif m == 'a':
                    a_mask = h_rec
                    recon_mse = recon_mse + F.mse_loss(a_mask, gated_full_a)
                else:
                    f_mask = h_rec
                    recon_mse = recon_mse + F.mse_loss(f_mask, gated_full_f)
                exist_feats.append(h_rec)

            z_recon = self.bottleneck([v_mask, a_mask, f_mask])
            joint_loss = F.mse_loss(z_recon, z_full)

            recon_loss = recon_mse + beta_energy * energy_term + gamma_joint * joint_loss

        # Generate expert features
        expert_vv, expert_va, expert_vf = self.experts['v'](v_mask.transpose(1, 2)).transpose(1, 2), self.experts['a'](v_mask.transpose(1, 2)).transpose(1, 2), self.experts['f'](v_mask.transpose(1, 2)).transpose(1, 2)
        expert_av, expert_aa, expert_af = self.experts['v'](a_mask.transpose(1, 2)).transpose(1, 2), self.experts['a'](a_mask.transpose(1, 2)).transpose(1, 2), self.experts['f'](a_mask.transpose(1, 2)).transpose(1, 2)
        expert_fv, expert_fa, expert_ff = self.experts['v'](f_mask.transpose(1, 2)).transpose(1, 2), self.experts['a'](f_mask.transpose(1, 2)).transpose(1, 2), self.experts['f'](f_mask.transpose(1, 2)).transpose(1, 2)

        # Calculate routing weights
        router_weight_v = F.softmax(self.router_mix(v_mask), dim=-1)
        router_weight_a = F.softmax(self.router_mix(a_mask), dim=-1)
        router_weight_f = F.softmax(self.router_mix(f_mask), dim=-1)

        # Feature aggregation
        fused_v = torch.cat([expert_vv.unsqueeze(-2), expert_va.unsqueeze(-2), expert_vf.unsqueeze(-2)],dim=-2)
        fusion_v = (fused_v * router_weight_v.unsqueeze(-1).repeat(1, 1, 1, self.hidden_dim)).sum(dim=2)
        fused_a = torch.cat([expert_av.unsqueeze(-2), expert_aa.unsqueeze(-2), expert_af.unsqueeze(-2)],dim=-2)
        fusion_a = (fused_a * router_weight_a.unsqueeze(-1).repeat(1, 1, 1, self.hidden_dim)).sum(dim=2)
        fused_f = torch.cat([expert_fv.unsqueeze(-2), expert_fa.unsqueeze(-2), expert_ff.unsqueeze(-2)],dim=-2)
        fusion_f = (fused_f * router_weight_f.unsqueeze(-1).repeat(1, 1, 1, self.hidden_dim)).sum(dim=2)
        fusion = torch.cat([fusion_v, fusion_a, fusion_f], -1)
        fusion = self.fusion(fusion.transpose(1, 2)).transpose(1, 2)
        # Decoding process
        prototype = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        prototype = self.pos_encoder(prototype)
        fea, att_weights = self.transformer.decoder(prototype, fusion)
        s = self.regressor(fea)  # (b, n, n)
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n)
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        
        # Generate expert features (for KL loss) - use full-context gated features for consistency
        expert_v = self.experts['v'](gated_full_v.transpose(1, 2)).transpose(1, 2)
        expert_a = self.experts['a'](gated_full_a.transpose(1, 2)).transpose(1, 2)
        expert_f = self.experts['f'](gated_full_f.transpose(1, 2)).transpose(1, 2)
        recon_loss += F.kl_div(torch.log_softmax(fusion_v, -1), torch.softmax(expert_v, -1))
        recon_loss += F.kl_div(torch.log_softmax(fusion_a, -1), torch.softmax(expert_a, -1))
        recon_loss += F.kl_div(torch.log_softmax(fusion_f, -1), torch.softmax(expert_f, -1))

        # Multi-QuAD Δ 近似：q 的 L1 正则，迫使门控稀疏（论文 ℒ^s 精神）
        if gate_info is None:
            gate_loss = torch.tensor(0.0, device=proj_v.device, dtype=proj_v.dtype)
        else:
            gate_loss = (gate_info['q_v'].mean() + gate_info['q_a'].mean() + gate_info['q_f'].mean()) / 3.0

        return {'output': out, 'embed': fea, 'embed2': None, "recon_loss": recon_loss, "gate_loss": gate_loss}

    def forward_test(self, video, audio, flow, mask):
        b, t, c = video.shape
        
        v_pres, a_pres, f_pres = mask
        proj_v = torch.zeros(b, t, self.hidden_dim, device=video.device)
        proj_a = torch.zeros(b, t, self.hidden_dim, device=video.device)
        proj_f = torch.zeros(b, t, self.hidden_dim, device=video.device)

        if v_pres == 1:
            v = self.in_proj['v'](video.transpose(1, 2)).transpose(1, 2)
            proj_v = self.transformer.encoder(v)
        if a_pres == 1:
            a = self.in_proj['a'](audio.transpose(1, 2)).transpose(1, 2)
            proj_a = self.transformer.encoder(a)
        if f_pres == 1:
            f = self.in_proj['f'](flow.transpose(1, 2)).transpose(1, 2)
            proj_f = self.transformer.encoder(f)

        # LGP Gate (Multi-QuAD Eq.2)
        presence = torch.tensor([[v_pres, a_pres, f_pres]], device=proj_v.device,
                                 dtype=proj_v.dtype).expand(b, 3)
        if getattr(self.config, 'no_lgp', False):
            gated_v, gated_a, gated_f = proj_v, proj_a, proj_f
        else:
            gated_v, gated_a, gated_f, _ = self.lgp_gate(proj_v, proj_a, proj_f, presence)

        v_mask = gated_v
        a_mask = gated_a
        f_mask = gated_f

        # DCER Stage2+Stage3 reconstruction when modalities are missing
        exist_feats = []
        if v_pres == 1:
            exist_feats.append(v_mask)
        if a_pres == 1:
            exist_feats.append(a_mask)
        if f_pres == 1:
            exist_feats.append(f_mask)

        missing = []
        if v_pres == 0:
            missing.append('v')
        if a_pres == 0:
            missing.append('a')
        if f_pres == 0:
            missing.append('f')

        if len(missing) > 0 and not getattr(self.config, 'no_decr', False):
            z_obs = self.bottleneck(exist_feats)
            cfg = self.config
            ebr_steps = int(getattr(cfg, 'ebr_steps', 3))
            ebr_lr = float(getattr(cfg, 'ebr_lr', 0.1))
            ebr_momentum = float(getattr(cfg, 'ebr_momentum', 0.0))
            # Inference defaults to non-unrolled to reduce memory.
            ebr_unroll = bool(getattr(cfg, 'ebr_unroll_test', False))

            for m in missing:
                z_obs = self.bottleneck(exist_feats)
                # NOTE: test.py wraps forward in torch.no_grad(); energy-based reconstruction
                # requires gradients, so we locally re-enable grad for this block.
                with torch.enable_grad():
                    h_rec, _ = self.ebr[m](exist_feats, z_obs, n_steps=ebr_steps, step_size=ebr_lr,
                                           momentum=ebr_momentum, unroll=ebr_unroll)
                h_rec = h_rec.detach()
                if m == 'v':
                    v_mask = h_rec
                elif m == 'a':
                    a_mask = h_rec
                else:
                    f_mask = h_rec
                exist_feats.append(h_rec)

        # Generate expert features
        expert_vv, expert_va, expert_vf = self.experts['v'](v_mask.transpose(1, 2)).transpose(1, 2), self.experts['a'](v_mask.transpose(1, 2)).transpose(1, 2), self.experts['f'](v_mask.transpose(1, 2)).transpose(1, 2)
        expert_av, expert_aa, expert_af = self.experts['v'](a_mask.transpose(1, 2)).transpose(1, 2), self.experts['a'](a_mask.transpose(1, 2)).transpose(1, 2), self.experts['f'](a_mask.transpose(1, 2)).transpose(1, 2)
        expert_fv, expert_fa, expert_ff = self.experts['v'](f_mask.transpose(1, 2)).transpose(1, 2), self.experts['a'](f_mask.transpose(1, 2)).transpose(1, 2), self.experts['f'](f_mask.transpose(1, 2)).transpose(1, 2)

        # Calculate routing weights
        router_weight_v = F.softmax(self.router_mix(v_mask), dim=-1)
        router_weight_a = F.softmax(self.router_mix(a_mask), dim=-1)
        router_weight_f = F.softmax(self.router_mix(f_mask), dim=-1)

        # Feature aggregation
        fused_v = torch.cat([expert_vv.unsqueeze(-2), expert_va.unsqueeze(-2), expert_vf.unsqueeze(-2)],dim=-2)
        fusion_v = (fused_v * router_weight_v.unsqueeze(-1).repeat(1, 1, 1, self.hidden_dim)).sum(dim=2)
        fused_a = torch.cat([expert_av.unsqueeze(-2), expert_aa.unsqueeze(-2), expert_af.unsqueeze(-2)],dim=-2)
        fusion_a = (fused_a * router_weight_a.unsqueeze(-1).repeat(1, 1, 1, self.hidden_dim)).sum(dim=2)
        fused_f = torch.cat([expert_fv.unsqueeze(-2), expert_fa.unsqueeze(-2), expert_ff.unsqueeze(-2)],dim=-2)
        fusion_f = (fused_f * router_weight_f.unsqueeze(-1).repeat(1, 1, 1, self.hidden_dim)).sum(dim=2)
        fusion = torch.cat([fusion_v, fusion_a, fusion_f], -1)
        fusion = self.fusion(fusion.transpose(1, 2)).transpose(1, 2)
        # Decoding process
        prototype = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        prototype = self.pos_encoder(prototype)
        fea, att_weights = self.transformer.decoder(prototype, fusion)
        s = self.regressor(fea)  # (b, n, n)
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n)
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        return {'output': out, 'embed': fea, 'fusion': fusion}
