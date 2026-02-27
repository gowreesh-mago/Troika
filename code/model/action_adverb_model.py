import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.common import MLP, QuickGELU


class Disentangler(nn.Module):
    """Disentangler module from Troika"""
    def __init__(self, emb_dim):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


class MulitHeadAttention(nn.Module):
    """Multi-head attention from Troika"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer from Troika"""
    def __init__(self, d_model, nhead, dropout=0.1,):
        super().__init__()
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, q, kv):
        q = q + self.cross_attn(q, kv, kv)
        q = q + self.dropout(self.mlp(self.norm(q)))
        return q


class GloVeEmbeddingLayer(nn.Module):
    """Manages GloVe embeddings with learnable projection"""
    def __init__(self, vocab, glove_path, embedding_dim=300, hidden_dim=768, freeze=False):
        super().__init__()
        self.vocab = vocab
        self.vocab_to_idx = {w.lower(): i for i, w in enumerate(vocab)}
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Initialize embeddings from GloVe
        embeddings = self._load_glove_embeddings(vocab, glove_path, embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=freeze)

        # Learnable projection to hidden_dim
        self.projection = nn.Linear(embedding_dim, hidden_dim)

    def _load_glove_embeddings(self, vocab, glove_path, embedding_dim):
        """Load GloVe embeddings for vocabulary words"""
        vocab_lower = sorted([w.lower() for w in vocab])
        vocab_to_idx = {w: i for i, w in enumerate(vocab_lower)}

        # Initialize with small random values
        embeddings = np.random.randn(len(vocab), embedding_dim).astype(np.float32) * 0.01

        found = set()
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                word = tokens[0]
                if word in vocab_to_idx:
                    vec = np.array(tokens[1:], dtype=np.float32)
                    embeddings[vocab_to_idx[word]] = vec
                    found.add(word)

        print(f"GloVe embeddings: {len(found)}/{len(vocab)} words found")
        not_found = list(set(vocab_lower) - found)
        if not_found:
            print(f"Not found in GloVe: {not_found[:10]}{'...' if len(not_found) > 10 else ''}")

        return torch.FloatTensor(embeddings)

    def forward(self, indices):
        """
        Args:
            indices: (batch_size,) or (num_words,) tensor of word indices

        Returns:
            Normalized projected embeddings: (*, hidden_dim)
        """
        emb = self.embedding(indices)  # (*, embedding_dim)
        proj = self.projection(emb)    # (*, hidden_dim)
        return F.normalize(proj, dim=-1)


class VisualFeatureEncoder(nn.Module):
    """Encodes pre-extracted visual features (flow or RGB) with temporal dimension"""
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        # Reuse MLP from common.py
        self.encoder = MLP(
            inp_dim=input_dim,
            out_dim=hidden_dim,
            num_layers=3,
            relu=True,
            dropout=True,
            norm=True
        )

    def forward(self, x):
        """
        Args:
            x: (B, L, C) temporal features or (B, C) single-frame features
        Returns:
            (B, hidden_dim) encoded features
        """
        if x.dim() == 3:  # (B, L, C) - temporal features
            # Apply temporal pooling (mean over time)
            x = x.mean(dim=1)  # (B, C)
        return self.encoder(x)


class ActionAdverbModel(nn.Module):
    """
    Action-Adverb classification model using pre-extracted I3D features and GloVe embeddings.
    Adapted from Troika's CZSL architecture.
    """
    def __init__(self, config, action_vocab, adverb_vocab, flow_dim, rgb_dim):
        super().__init__()
        self.config = config
        self.action_vocab = action_vocab
        self.adverb_vocab = adverb_vocab

        hidden_dim = config.hidden_dim

        # Visual encoders for flow and RGB features
        self.flow_encoder = VisualFeatureEncoder(flow_dim, hidden_dim, config.dropout)
        self.rgb_encoder = VisualFeatureEncoder(rgb_dim, hidden_dim, config.dropout)

        # Feature fusion MLP
        self.fusion_mlp = MLP(
            inp_dim=hidden_dim * 2,
            out_dim=hidden_dim,
            num_layers=2,
            dropout=True,
            norm=True
        )

        # Disentanglers for action and adverb features (adapted from Troika)
        self.action_disentangler = Disentangler(hidden_dim)
        self.adverb_disentangler = Disentangler(hidden_dim)

        # Text embeddings using GloVe
        self.action_embeddings = GloVeEmbeddingLayer(
            action_vocab,
            config.glove_path,
            300,
            hidden_dim,
            freeze=config.freeze_glove
        )
        self.adverb_embeddings = GloVeEmbeddingLayer(
            adverb_vocab,
            config.glove_path,
            300,
            hidden_dim,
            freeze=config.freeze_glove
        )

        # Cross-modal traction layers (adapted from Troika's CMT)
        self.cmt_action = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, config.num_heads, config.cmt_dropout)
            for _ in range(config.cmt_layers)
        ])
        self.cmt_adverb = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, config.num_heads, config.cmt_dropout)
            for _ in range(config.cmt_layers)
        ])

        # Learnable blending weights (like lambda in Troika)
        self.lamda_action = nn.Parameter(torch.ones(1) * config.init_lamda)
        self.lamda_adverb = nn.Parameter(torch.ones(1) * config.init_lamda)

        # Temperature for similarity scaling
        self.temperature = config.temperature

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Store all pairs for evaluation
        self.all_pairs = [(a, av) for a in action_vocab for av in adverb_vocab]

        print(f'\nActionAdverbModel initialized:')
        print(f'  - Actions: {len(action_vocab)}, Adverbs: {len(adverb_vocab)}')
        print(f'  - Total pairs: {len(self.all_pairs)}')
        print(f'  - Flow dim: {flow_dim}, RGB dim: {rgb_dim}, Hidden dim: {hidden_dim}')
        print(f'  - CMT layers: {config.cmt_layers}, Attention heads: {config.num_heads}')
        print(f'  - Temperature: {config.temperature}\n')

    def encode_visual(self, flow_features, rgb_features):
        """
        Encode and fuse flow and RGB features.

        Args:
            flow_features: (B, flow_dim)
            rgb_features: (B, rgb_dim)

        Returns:
            Fused visual features: (B, hidden_dim)
        """
        flow_encoded = self.flow_encoder(flow_features)  # (B, hidden_dim)
        rgb_encoded = self.rgb_encoder(rgb_features)      # (B, hidden_dim)

        # Late fusion: concatenate then MLP
        fused = torch.cat([flow_encoded, rgb_encoded], dim=-1)  # (B, 2*hidden_dim)
        visual_feats = self.fusion_mlp(fused)  # (B, hidden_dim)

        return visual_feats

    def forward(self, batch, all_action_indices, all_adverb_indices):
        """
        Forward pass for training/evaluation.

        Args:
            batch: Dict with keys:
                - 'flow_features': (B, flow_dim)
                - 'rgb_features': (B, rgb_dim)
                - 'action_idx': (B,)
                - 'adverb_idx': (B,)
                - 'pair_idx': (B,)
            all_action_indices: Tensor of all action indices (num_actions,)
            all_adverb_indices: Tensor of all adverb indices (num_adverbs,)

        Returns:
            List of [pair_logits, action_logits, adverb_logits]
                - pair_logits: (B, num_pairs)
                - action_logits: (B, num_actions)
                - adverb_logits: (B, num_adverbs)
        """
        # Get device from model parameters
        device = next(self.parameters()).device
        flow_feats = batch['flow_features'].to(device)
        rgb_feats = batch['rgb_features'].to(device)

        # 1. Encode visual features
        visual_feats = self.encode_visual(flow_feats, rgb_feats)  # (B, hidden_dim)

        # 2. Disentangle into action and adverb specific features
        visual_action = self.action_disentangler(visual_feats)  # (B, hidden_dim)
        visual_adverb = self.adverb_disentangler(visual_feats)  # (B, hidden_dim)

        # 3. Get text embeddings for all actions and adverbs
        action_text_emb = self.action_embeddings(all_action_indices)  # (num_actions, hidden_dim)
        adverb_text_emb = self.adverb_embeddings(all_adverb_indices)  # (num_adverbs, hidden_dim)

        # 4. Apply Cross-Modal Traction (CMT)
        B = visual_action.shape[0]

        # Action branch: text embeddings attend to visual features
        visual_action_kv = visual_action.unsqueeze(1)  # (B, 1, hidden_dim)
        action_text_q = action_text_emb.unsqueeze(0).expand(B, -1, -1)  # (B, num_actions, hidden_dim)

        for layer in self.cmt_action:
            action_text_q = layer(action_text_q, visual_action_kv)

        # Blend original text embeddings with enhanced ones
        action_text_enhanced = action_text_emb + self.lamda_action * action_text_q.mean(0)

        # Adverb branch (similar process)
        visual_adverb_kv = visual_adverb.unsqueeze(1)  # (B, 1, hidden_dim)
        adverb_text_q = adverb_text_emb.unsqueeze(0).expand(B, -1, -1)  # (B, num_adverbs, hidden_dim)

        for layer in self.cmt_adverb:
            adverb_text_q = layer(adverb_text_q, visual_adverb_kv)

        adverb_text_enhanced = adverb_text_emb + self.lamda_adverb * adverb_text_q.mean(0)

        # 5. Normalize features for cosine similarity
        visual_action_norm = F.normalize(visual_action, dim=-1)
        visual_adverb_norm = F.normalize(visual_adverb, dim=-1)
        action_text_norm = F.normalize(action_text_enhanced, dim=-1)
        adverb_text_norm = F.normalize(adverb_text_enhanced, dim=-1)

        # 6. Compute logits (cosine similarity with temperature scaling)
        action_logits = (visual_action_norm @ action_text_norm.T) * self.temperature  # (B, num_actions)
        adverb_logits = (visual_adverb_norm @ adverb_text_norm.T) * self.temperature  # (B, num_adverbs)

        # 7. Compute pair logits by combining action and adverb scores
        # For each (action, adverb) pair, sum their individual logits
        pair_logits = []
        for (act, adv) in self.all_pairs:
            act_idx = self.action_vocab.index(act)
            adv_idx = self.adverb_vocab.index(adv)
            pair_score = action_logits[:, act_idx] + adverb_logits[:, adv_idx]  # (B,)
            pair_logits.append(pair_score)
        pair_logits = torch.stack(pair_logits, dim=1)  # (B, num_pairs)

        return [pair_logits, action_logits, adverb_logits]

    def loss_calu(self, predict, batch):
        """
        Calculate weighted multi-branch loss (same as Troika's training objective).

        Args:
            predict: List of [pair_logits, action_logits, adverb_logits]
            batch: Dict with ground truth labels

        Returns:
            Total weighted loss (scalar)
        """
        pair_logits, action_logits, adverb_logits = predict

        # Get device from model parameters
        device = next(self.parameters()).device
        action_labels = batch['action_idx'].to(device)
        adverb_labels = batch['adverb_idx'].to(device)
        pair_labels = batch['pair_idx'].to(device)

        # Compute individual losses
        loss_pair = self.loss_fn(pair_logits, pair_labels)
        loss_action = self.loss_fn(action_logits, action_labels)
        loss_adverb = self.loss_fn(adverb_logits, adverb_labels)

        # Weighted combination (same structure as Troika)
        total_loss = (loss_pair * self.config.pair_loss_weight +
                     loss_action * self.config.action_loss_weight +
                     loss_adverb * self.config.adverb_loss_weight)

        return total_loss
