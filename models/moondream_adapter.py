"""
White-box differentiable adapter for HuggingFace moondream2.

Provides gradient-compatible access to three attack surfaces:
  Level 1 — vision encoder features  (pre-projection)
  Level 2 — connector tokens          (vision projection output)
  Level 3 — LLM logits                (full pipeline)

All differentiable methods accept a [1, 3, H, W] float32 tensor
in [0, 1] and preserve the autograd graph back to the input image.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM


class MoondreamAdapter:

    MODEL_NAME = "vikhyatk/moondream2"
    CROP_SIZE = 378
    PATCH_SIZE = 14
    N_PATCHES = 27 * 27  # 729
    PREFIX_LEN = 1 + 27 * 27  # 730 (BOS + image tokens)

    def __init__(self, device=None, dtype=torch.bfloat16):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype
        self.name = "moondream2-hf"

        # --------------------------------------------------------
        # Load model
        # --------------------------------------------------------

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)

        self.model.eval()

        # Shortcut to internal MoondreamModel
        self._mm = self.model.model
        self._config = self._mm.config

        # --------------------------------------------------------
        # Fix: transformers v5 meta-device loading wipes non-persistent
        # buffers (freqs_cis, attn_mask) to zeros. Re-initialize them.
        # --------------------------------------------------------
        self._restore_buffers()

        # Setup KV caches for built-in query()/caption() methods
        self.model._setup_caches()

        # Precompute prompt tokens for the standard query
        self._prompt_tokens = self._build_prompt_tokens(
            "What do you see in this image?"
        )

    # ============================================================
    # QUANTIZATION SIMULATION (for transfer to Q4 Ollama)
    # ============================================================

    @staticmethod
    def _quantize_q4_0_ste(weight, block_size=32):
        """
        Simulate Q4_0 quantization with straight-through estimator.

        Q4_0: weights grouped into blocks of 32, each block has a
        scale factor (max_abs / 8) and 4-bit signed integers (-8..7).

        Forward: quantize -> dequantize (loses precision)
        Backward: gradient passes through unchanged (STE)
        """
        # Detach for quantization, keep original for STE
        w = weight.detach()
        orig = weight

        # Reshape into blocks
        orig_shape = w.shape
        w_flat = w.reshape(-1)
        n = w_flat.numel()
        pad = (block_size - n % block_size) % block_size
        if pad > 0:
            w_flat = F.pad(w_flat, (0, pad))
        w_blocks = w_flat.reshape(-1, block_size)

        # Q4_0: scale = max_abs / 8, quantized = round(w / scale), clamped to [-8, 7]
        max_abs = w_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = max_abs / 8.0
        q = torch.round(w_blocks / scale).clamp(-8, 7)
        w_q = q * scale

        # Reshape back
        w_q = w_q.reshape(-1)[:n].reshape(orig_shape)

        # Straight-through estimator: forward uses quantized, backward uses original
        return orig + (w_q - w).detach()

    def _apply_q4_to_module(self, module, enabled=True):
        """
        Monkey-patch all Linear layers in module to simulate Q4_0.
        Stores original forward methods for restoration.
        """
        if not hasattr(self, '_q4_patched'):
            self._q4_patched = []

        if not enabled:
            # Restore originals
            for layer, orig_forward in self._q4_patched:
                layer.forward = orig_forward
            self._q4_patched = []
            return

        import types

        for name, child in module.named_modules():
            if isinstance(child, torch.nn.Linear):
                # Store original
                self._q4_patched.append((child, child.forward))

                # Create quantized forward
                def make_q4_forward(orig_weight):
                    def q4_forward(self, x):
                        w_q = MoondreamAdapter._quantize_q4_0_ste(self.weight)
                        return F.linear(x, w_q, self.bias)
                    return q4_forward

                child.forward = types.MethodType(
                    make_q4_forward(child.weight), child
                )

    def _simulate_q4_noise(self, module, noise_scale=0.05):
        """
        Context manager: apply Q4_0 quantization simulation to all
        Linear layers. When noise_scale='q4', use proper Q4 simulation.
        Otherwise use Gaussian noise (legacy).
        """
        import contextlib

        if noise_scale == 'q4':
            self._apply_q4_to_module(module, enabled=True)

            @contextlib.contextmanager
            def ctx():
                try:
                    yield
                finally:
                    self._apply_q4_to_module(module, enabled=False)
        else:
            # Legacy Gaussian noise
            original_weights = {}
            for name, param in module.named_parameters():
                if 'weight' in name and param.ndim >= 2:
                    original_weights[name] = param.data.clone()
                    noise = torch.randn_like(param.data) * noise_scale * param.data.std()
                    param.data = param.data + noise

            @contextlib.contextmanager
            def ctx():
                try:
                    yield
                finally:
                    for name, param in module.named_parameters():
                        if name in original_weights:
                            param.data = original_weights[name]

        return ctx()

    # ============================================================
    # BUFFER RESTORATION (transformers v5 fix)
    # ============================================================

    @staticmethod
    def _precompute_freqs_cis(dim, end, theta=10000.0):
        freqs = 1.0 / (
            theta
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim
            )
        )
        t = torch.arange(end, dtype=torch.float32).unsqueeze(1)
        freqs = t * freqs.unsqueeze(0)
        freqs = torch.exp(1j * freqs)
        return torch.stack([freqs.real, freqs.imag], dim=-1)

    def _restore_buffers(self):
        c_text = self._config.text
        c_vis = self._config.vision

        self._mm.text.freqs_cis = self._precompute_freqs_cis(
            c_text.dim // (2 * c_text.n_heads), c_text.max_context
        ).to(self.device)

        attn_mask = torch.tril(
            torch.ones(
                1, 1, c_text.max_context, c_text.max_context, dtype=torch.bool
            )
        )
        patch_w = c_vis.crop_size // c_vis.enc_patch_size
        prefix_attn_len = 1 + patch_w**2
        attn_mask[..., :prefix_attn_len, :prefix_attn_len] = 1
        self._mm.attn_mask = attn_mask.to(self.device)

    # ============================================================
    # TOKENIZER HELPERS
    # ============================================================

    def _build_prompt_tokens(self, question):
        prefix = self._config.tokenizer.templates["query"]["prefix"]
        suffix = self._config.tokenizer.templates["query"]["suffix"]
        question_ids = self._mm.tokenizer.encode(question).ids
        tokens = prefix + question_ids + suffix + suffix
        return torch.tensor([tokens], device=self.device, dtype=torch.long)

    def encode_text(self, text):
        ids = self._mm.tokenizer.encode(text).ids
        return torch.tensor([ids], device=self.device, dtype=torch.long)

    def decode_tokens(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return self._mm.tokenizer.decode(ids)

    # ============================================================
    # DIFFERENTIABLE PREPROCESSING
    # ============================================================

    def _preprocess(self, image):
        """
        Differentiable single-crop preprocessing.

        Input:  [1, 3, H, W] float32 in [0, 1]
        Output: [1, 3, 378, 378] in [-1, 1], model dtype
        """
        x = F.interpolate(
            image,
            size=(self.CROP_SIZE, self.CROP_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - 0.5) / 0.5
        x = x.to(dtype=self.dtype)
        return x

    # ============================================================
    # REIMPLEMENTED INTERNAL FUNCTIONS
    # (pure functions, no torch.inference_mode, no KV cache)
    # ============================================================

    @staticmethod
    def _layer_norm(x, ln_module):
        return ln_module(x)

    @staticmethod
    def _mlp(x, mlp_module):
        return mlp_module.fc2(
            F.gelu(mlp_module.fc1(x), approximate="tanh")
        )

    @staticmethod
    def _vision_attn(x, attn_module, n_heads):
        bsz, q_len, d_model = x.shape
        head_dim = d_model // n_heads
        qkv = attn_module.qkv(x)
        q, k, v = [
            t.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
            for t in qkv.chunk(3, dim=-1)
        ]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(bsz, q_len, d_model)
        return attn_module.proj(out)

    def _vision_encoder(self, x):
        """Run vision encoder. Input: [1, 3, 378, 378] normalized."""
        cfg = self._config.vision
        w = self._mm.vision

        B, C, H, W = x.shape
        P = cfg.enc_patch_size

        patches = x.reshape(B, C, H // P, P, W // P, P)
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(
            B, (H // P) * (W // P), C * P * P
        )

        h = w.patch_emb(patches)
        h = h + w.pos_emb

        for block in w.blocks:
            h = h + self._vision_attn(
                self._layer_norm(h, block.ln1),
                block.attn,
                cfg.enc_n_heads,
            )
            h = h + self._mlp(
                self._layer_norm(h, block.ln2),
                block.mlp,
            )

        h = self._layer_norm(h, w.post_ln)
        return h  # [1, 729, 1152]

    def _vision_projection(self, global_features, local_features):
        """Project vision features to LLM dimension."""
        cfg = self._config.vision
        w = self._mm.vision

        reconstructed = local_features.view(
            cfg.enc_n_layers, cfg.enc_n_layers, cfg.enc_dim
        )
        reconstructed = reconstructed.permute(2, 0, 1)
        reconstructed = F.adaptive_avg_pool2d(
            reconstructed,
            output_size=(cfg.enc_n_layers, cfg.enc_n_layers),
        )
        reconstructed = reconstructed.permute(1, 2, 0).view(
            self.N_PATCHES, cfg.enc_dim
        )

        final = torch.cat([global_features, reconstructed], dim=-1)
        return self._mlp(final, w.proj_mlp)  # [729, 2048]

    @staticmethod
    def _apply_rope(x, freqs_cis, position_ids, num_heads, rot_dim=32):
        """Apply rotary position embeddings."""
        x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        d_q = x_rot.shape[-1] // 2
        xq_r, xq_i = x_rot[..., :d_q], x_rot[..., d_q:]

        freqs_cos = (
            freqs_cis[..., 0][position_ids, :]
            .unsqueeze(0)
            .unsqueeze(0)
        )
        freqs_sin = (
            freqs_cis[..., 1][position_ids, :]
            .unsqueeze(0)
            .unsqueeze(0)
        )

        xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
        xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
        xq_out = torch.stack(
            (xq_out_r, xq_out_i), dim=-1
        ).flatten(-2)

        return torch.cat(
            [xq_out.to(x.dtype), x_pass], dim=-1
        )

    def _text_attn(self, x, attn_module, position_ids, attn_mask):
        """Text attention with RoPE, no KV cache."""
        cfg = self._config.text
        freqs_cis = self._mm.text.freqs_cis

        bsz, q_len, d_model = x.shape
        head_dim = d_model // cfg.n_heads

        qkv_out = attn_module.qkv(x)
        q_dim = cfg.n_heads * head_dim
        kv_dim = cfg.n_kv_heads * head_dim
        q, k, v = qkv_out.split([q_dim, kv_dim, kv_dim], dim=-1)

        q = q.view(bsz, q_len, cfg.n_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, cfg.n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, cfg.n_kv_heads, head_dim).transpose(1, 2)

        q = self._apply_rope(q, freqs_cis, position_ids, cfg.n_heads)
        k = self._apply_rope(k, freqs_cis, position_ids, cfg.n_kv_heads)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            enable_gqa=cfg.n_heads != cfg.n_kv_heads,
        )
        out = out.transpose(1, 2).reshape(bsz, q_len, d_model)
        return attn_module.proj(out)

    def _text_decoder(self, x, attn_mask, position_ids):
        """Text decoder without KV cache."""
        w = self._mm.text

        for block in w.blocks:
            l_in = self._layer_norm(x, block.ln)
            l_attn = self._text_attn(
                l_in, block.attn, position_ids, attn_mask
            )
            l_mlp = self._mlp(l_in, block.mlp)
            x = x + l_attn + l_mlp

        return x

    def _lm_head(self, hidden):
        """LM head: layer norm + linear projection on last token."""
        w = self._mm.text
        hidden_last = hidden[:, -1, :]
        hidden_last = self._layer_norm(hidden_last, w.post_ln)
        return w.lm_head(hidden_last)

    def _lm_head_all(self, hidden):
        """LM head applied to ALL positions (not just last)."""
        w = self._mm.text
        hidden_normed = self._layer_norm(hidden, w.post_ln)
        return w.lm_head(hidden_normed)

    def _build_attn_mask(self, q_len, prefix_len=None):
        """
        Build attention mask:
        - bidirectional for the first prefix_len tokens (BOS + image)
        - causal for the remaining tokens (prompt)
        """
        if prefix_len is None:
            prefix_len = self.PREFIX_LEN

        mask = torch.zeros(
            1, 1, q_len, q_len, dtype=torch.bool, device=self.device
        )
        mask[:, :, :prefix_len, :prefix_len] = True
        for i in range(prefix_len, q_len):
            mask[:, :, i, : i + 1] = True
        return mask

    # ============================================================
    # PUBLIC DIFFERENTIABLE METHODS
    # ============================================================

    def get_vision_features(self, image):
        """
        Level 1: Vision encoder features (before projection).

        Input:  [1, 3, H, W] float32 in [0, 1]
        Output: [1, 729, 1152]
        """
        x = self._preprocess(image)
        return self._vision_encoder(x)

    def get_connector_tokens(self, image):
        """
        Level 2: Connector tokens (vision projection output).

        Input:  [1, 3, H, W] float32 in [0, 1]
        Output: [1, 729, 2048]
        """
        x = self._preprocess(image)
        features = self._vision_encoder(x)

        global_features = features[0]
        local_features = features[0]

        tokens = self._vision_projection(global_features, local_features)
        return tokens.unsqueeze(0)

    def get_llm_logits(self, image, prompt_tokens=None):
        """
        Level 3: LLM logits after full pipeline.

        Input:  [1, 3, H, W] float32 in [0, 1]
        Output: [1, vocab_size] logits from last position
        """
        if prompt_tokens is None:
            prompt_tokens = self._prompt_tokens

        x = self._preprocess(image)
        features = self._vision_encoder(x)

        global_features = features[0]
        local_features = features[0]
        img_tokens = self._vision_projection(
            global_features, local_features
        )

        cfg = self._config
        w = self._mm.text

        bos_emb = F.embedding(
            torch.tensor(
                [[cfg.tokenizer.bos_id]],
                device=self.device,
                dtype=torch.long,
            ),
            w.wte,
        )

        prompt_emb = F.embedding(prompt_tokens, w.wte)

        inputs_embeds = torch.cat(
            [bos_emb, img_tokens.unsqueeze(0), prompt_emb], dim=1
        )

        q_len = inputs_embeds.size(1)
        attn_mask = self._build_attn_mask(q_len)
        position_ids = torch.arange(
            q_len, device=self.device, dtype=torch.long
        )

        hidden = self._text_decoder(inputs_embeds, attn_mask, position_ids)
        logits = self._lm_head(hidden)
        return logits

    # ============================================================
    # DIFFERENTIABLE MULTI-CROP PIPELINE
    # ============================================================

    @staticmethod
    def _select_tiling(height, width, crop_size, max_crops):
        """Determine optimal tiling for multi-crop (pure math, no gradient)."""
        import math

        if height <= crop_size or width <= crop_size:
            return (1, 1)

        min_h = math.ceil(height / crop_size)
        min_w = math.ceil(width / crop_size)

        if min_h * min_w > max_crops:
            ratio = math.sqrt(max_crops / (min_h * min_w))
            return (
                max(1, math.floor(min_h * ratio)),
                max(1, math.floor(min_w * ratio)),
            )

        h_tiles = math.floor(math.sqrt(max_crops * height / width))
        w_tiles = math.floor(math.sqrt(max_crops * width / height))
        h_tiles = max(h_tiles, min_h)
        w_tiles = max(w_tiles, min_w)

        if h_tiles * w_tiles > max_crops:
            if w_tiles > h_tiles:
                w_tiles = math.floor(max_crops / h_tiles)
            else:
                h_tiles = math.floor(max_crops / w_tiles)

        return (max(1, h_tiles), max(1, w_tiles))

    def _diff_overlap_crop(self, image):
        """
        Differentiable overlap_crop_image using torch operations.

        Input:  [1, 3, H, W] float32 in [0, 1]
        Output: (crops [N, 3, 378, 378] float32 in [-1, 1], tiling (h, w))

        Crops[0] is the global crop (full image resized to 378x378).
        Crops[1:] are local overlapping crops.
        """
        import math

        cfg = self._config.vision
        crop_size = cfg.crop_size  # 378
        patch_size = cfg.enc_patch_size  # 14
        overlap_margin = cfg.overlap_margin  # 4
        max_crops = cfg.max_crops  # 12

        margin_pixels = patch_size * overlap_margin  # 56
        total_margin = margin_pixels * 2  # 112

        crop_patches = crop_size // patch_size  # 27
        crop_window_patches = crop_patches - (2 * overlap_margin)  # 19
        crop_window_size = crop_window_patches * patch_size  # 266

        _, _, H, W = image.shape

        tiling = self._select_tiling(
            H - total_margin, W - total_margin, crop_window_size, max_crops
        )

        target_h = tiling[0] * crop_window_size + total_margin
        target_w = tiling[1] * crop_window_size + total_margin

        # Differentiable resize (bilinear, matches LANCZOS approximately)
        resized = F.interpolate(
            image, size=(target_h, target_w), mode="bilinear",
            align_corners=False, antialias=True,
        )

        # Global crop: resize original to crop_size x crop_size
        global_crop = F.interpolate(
            image, size=(crop_size, crop_size), mode="bilinear",
            align_corners=False, antialias=True,
        )

        n_crops = tiling[0] * tiling[1] + 1
        crops = torch.zeros(
            n_crops, 3, crop_size, crop_size,
            device=image.device, dtype=image.dtype,
        )
        crops[0] = global_crop[0]

        for i in range(tiling[0]):
            for j in range(tiling[1]):
                y0 = i * crop_window_size
                x0 = j * crop_window_size
                y_end = min(y0 + crop_size, target_h)
                x_end = min(x0 + crop_size, target_w)

                crop_region = resized[0, :, y0:y_end, x0:x_end]

                idx = 1 + i * tiling[1] + j
                crops[idx, :, : y_end - y0, : x_end - x0] = crop_region

        # Normalize to [-1, 1] (same as prepare_crops)
        crops = crops.sub_(0.5).div_(0.5)

        return crops, tiling

    def _diff_reconstruct(self, crops, tiling):
        """
        Differentiable reconstruct_from_crops (feature-level).

        Input:  crops [N, 27, 27, enc_dim] (local crops, HWC format)
                tiling (h, w)
        Output: [output_h, output_w, enc_dim] (HWC format)
        """
        overlap_margin = self._config.vision.overlap_margin
        patch_size = 1
        margin_pixels = overlap_margin * patch_size

        tiling_h, tiling_w = tiling
        crop_height, crop_width = crops[0].shape[:2]  # 27, 27
        enc_dim = crops[0].shape[2]  # 1152

        output_h = (crop_height - 2 * margin_pixels) * tiling_h + 2 * margin_pixels
        output_w = (crop_width - 2 * margin_pixels) * tiling_w + 2 * margin_pixels

        reconstructed = torch.zeros(
            output_h, output_w, enc_dim,
            device=crops.device, dtype=crops.dtype,
        )

        for i in range(tiling_h * tiling_w):
            tile_y = i // tiling_w
            tile_x = i % tiling_w

            x_start = 0 if tile_x == 0 else margin_pixels
            x_end = crop_width if tile_x == tiling_w - 1 else crop_width - margin_pixels
            y_start = 0 if tile_y == 0 else margin_pixels
            y_end = crop_height if tile_y == tiling_h - 1 else crop_height - margin_pixels

            out_x = tile_x * (crop_width - 2 * margin_pixels)
            out_y = tile_y * (crop_height - 2 * margin_pixels)

            crop = crops[i]
            reconstructed[
                out_y + y_start : out_y + y_end,
                out_x + x_start : out_x + x_end,
            ] = crop[y_start:y_end, x_start:x_end]

        return reconstructed

    def get_multicrop_tokens(self, image):
        """
        Differentiable multi-crop vision pipeline (matches model's
        _run_vision_encoder exactly).

        Input:  [1, 3, H, W] float32 in [0, 1]
        Output: [1, 729, 2048] connector tokens
        """
        crops, tiling = self._diff_overlap_crop(image)

        # Run vision encoder on all crops (batch)
        # crops: [N, 3, 378, 378] in [-1, 1]
        outputs = self._mm._vis_enc(crops.to(self.dtype))

        cfg = self._config.vision
        global_features = outputs[0]  # [enc_dim]
        local_features = outputs[1:].view(
            -1,
            cfg.enc_n_layers,
            cfg.enc_n_layers,
            cfg.enc_dim,
        )  # [n_local_crops, 27, 27, 1152] — HWC format (same as original)

        # Reconstruct local features (returns HWC)
        reconstructed = self._diff_reconstruct(local_features, tiling)

        # Projection (vision_projection handles adaptive_avg_pool2d to 27x27)
        img_tokens = self._mm._vis_proj(
            global_features, reconstructed
        )  # [729, 2048]

        return img_tokens.unsqueeze(0)

    def get_multicrop_multi_token_loss(
        self, image, target_ids, prompt_tokens=None, q4_noise=0.0
    ):
        """
        Multi-token CE loss through the FULL multi-crop pipeline.

        This is the loss that matches what HF describe() and Ollama see.

        Args:
            image:          [1, 3, H, W] float32 in [0, 1]
            target_ids:     [1, T] long tensor
            prompt_tokens:  [1, P] long tensor
            q4_noise:       If > 0, add Gaussian noise to weights
                            to simulate Q4 quantization (improves transfer)

        Returns:
            loss:   scalar CE loss
            logits: [1, T, vocab_size]
        """
        if prompt_tokens is None:
            prompt_tokens = self._prompt_tokens

        if q4_noise:
            ctx = self._simulate_q4_noise(self._mm, q4_noise)
        else:
            import contextlib
            ctx = contextlib.nullcontext()

        with ctx:
            img_tokens = self.get_multicrop_tokens(image)

            cfg = self._config
            w = self._mm.text

            bos_emb = F.embedding(
                torch.tensor(
                    [[cfg.tokenizer.bos_id]],
                    device=self.device,
                    dtype=torch.long,
                ),
                w.wte,
            )

            prompt_emb = F.embedding(prompt_tokens, w.wte)
            target_emb = F.embedding(target_ids, w.wte)

            inputs_embeds = torch.cat(
                [bos_emb, img_tokens, prompt_emb, target_emb], dim=1
            )

            q_len = inputs_embeds.size(1)
            attn_mask = self._build_attn_mask(q_len)
            position_ids = torch.arange(
                q_len, device=self.device, dtype=torch.long
            )

            hidden = self._text_decoder(inputs_embeds, attn_mask, position_ids)
            all_logits = self._lm_head_all(hidden)

            prefix_len = self.PREFIX_LEN
            prompt_len = prompt_tokens.size(1)
            target_len = target_ids.size(1)

            start = prefix_len + prompt_len - 1
            end = start + target_len
            target_logits = all_logits[:, start:end, :]

            loss = F.cross_entropy(
                target_logits.reshape(-1, target_logits.size(-1)),
                target_ids.reshape(-1),
            )

        return loss, target_logits

    def get_all_features(self, image, prompt_tokens=None):
        """
        All three levels in one forward pass.

        Returns dict with:
            vision_features:  [1, 729, 1152]
            connector_tokens: [1, 729, 2048]
            logits:           [1, vocab_size]
        """
        if prompt_tokens is None:
            prompt_tokens = self._prompt_tokens

        x = self._preprocess(image)
        features = self._vision_encoder(x)

        global_features = features[0]
        local_features = features[0]
        img_tokens = self._vision_projection(
            global_features, local_features
        )

        cfg = self._config
        w = self._mm.text

        bos_emb = F.embedding(
            torch.tensor(
                [[cfg.tokenizer.bos_id]],
                device=self.device,
                dtype=torch.long,
            ),
            w.wte,
        )

        prompt_emb = F.embedding(prompt_tokens, w.wte)

        inputs_embeds = torch.cat(
            [bos_emb, img_tokens.unsqueeze(0), prompt_emb], dim=1
        )

        q_len = inputs_embeds.size(1)
        attn_mask = self._build_attn_mask(q_len)
        position_ids = torch.arange(
            q_len, device=self.device, dtype=torch.long
        )

        hidden = self._text_decoder(inputs_embeds, attn_mask, position_ids)
        logits = self._lm_head(hidden)

        return {
            "vision_features": features,
            "connector_tokens": img_tokens.unsqueeze(0),
            "logits": logits,
        }

    def get_multi_token_loss(self, image, target_ids, prompt_tokens=None):
        """
        Multi-token targeted CE loss via teacher forcing.

        Feeds [BOS, img_tokens, prompt, target_tokens] through the
        full pipeline and computes CE loss at each target position.

        Args:
            image:          [1, 3, H, W] float32 in [0, 1]
            target_ids:     [1, T] long tensor of target token IDs
            prompt_tokens:  [1, P] long tensor of prompt token IDs

        Returns:
            loss:   scalar CE loss (mean across target positions)
            logits: [1, T, vocab_size] logits at target positions
        """
        if prompt_tokens is None:
            prompt_tokens = self._prompt_tokens

        x = self._preprocess(image)
        features = self._vision_encoder(x)

        global_features = features[0]
        local_features = features[0]
        img_tokens = self._vision_projection(
            global_features, local_features
        )

        cfg = self._config
        w = self._mm.text

        bos_emb = F.embedding(
            torch.tensor(
                [[cfg.tokenizer.bos_id]],
                device=self.device,
                dtype=torch.long,
            ),
            w.wte,
        )

        prompt_emb = F.embedding(prompt_tokens, w.wte)
        target_emb = F.embedding(target_ids, w.wte)

        inputs_embeds = torch.cat(
            [bos_emb, img_tokens.unsqueeze(0), prompt_emb, target_emb],
            dim=1,
        )

        q_len = inputs_embeds.size(1)
        attn_mask = self._build_attn_mask(q_len)
        position_ids = torch.arange(
            q_len, device=self.device, dtype=torch.long
        )

        hidden = self._text_decoder(inputs_embeds, attn_mask, position_ids)
        all_logits = self._lm_head_all(hidden)

        # Extract logits at positions that predict each target token
        # Position (prefix_len + prompt_len - 1) predicts target_tokens[0]
        # Position (prefix_len + prompt_len + i - 1) predicts target_tokens[i]
        prefix_len = self.PREFIX_LEN
        prompt_len = prompt_tokens.size(1)
        target_len = target_ids.size(1)

        start = prefix_len + prompt_len - 1
        end = start + target_len
        target_logits = all_logits[:, start:end, :]

        loss = F.cross_entropy(
            target_logits.reshape(-1, target_logits.size(-1)),
            target_ids.reshape(-1),
        )

        return loss, target_logits

    # ============================================================
    # NON-DIFFERENTIABLE EVALUATION
    # ============================================================

    @torch.no_grad()
    def describe(self, pil_image):
        """Generate a description using the model's built-in pipeline."""
        result = self.model.query(
            image=pil_image,
            question="What do you see in this image?",
            stream=False,
        )
        return result["answer"].strip()

    @torch.no_grad()
    def describe_single_crop(self, pil_image):
        """
        Generate a description using single-crop preprocessing.

        This matches the attack's preprocessing (F.interpolate to 378x378),
        so adversarial perturbations optimized for single-crop will be
        fully effective during evaluation.
        """
        import numpy as np
        import sys

        mod = sys.modules[self._mm.__class__.__module__]
        text_encoder_fn = mod.text_encoder
        EncodedImage = mod.EncodedImage

        # Convert PIL to tensor
        tensor = (
            torch.from_numpy(np.array(pil_image))
            .permute(2, 0, 1)
            .float()
            / 255.0
        )
        tensor = tensor.unsqueeze(0).to(self.device)

        # Single-crop preprocessing (same as attack)
        x = self._preprocess(tensor)

        # Run our vision encoder + projection
        features = self._vision_encoder(x)
        global_features = features[0]
        local_features = features[0]
        img_emb = self._vision_projection(
            global_features, local_features
        )  # [729, 2048]

        # Setup KV caches
        self.model._setup_caches()

        # BOS embedding using model's text_encoder
        cfg = self._config
        bos_emb = text_encoder_fn(
            torch.tensor(
                [[cfg.tokenizer.bos_id]],
                device=self.device,
                dtype=torch.long,
            ),
            self._mm.text,
        )

        # Concatenate BOS + image embeddings
        inputs_embeds = torch.cat(
            [bos_emb, img_emb[None]], dim=1
        )

        # Attention mask for prefix
        mask = self._mm.attn_mask[
            :, :, : inputs_embeds.size(1), :
        ]
        pos_ids = torch.arange(
            inputs_embeds.size(1),
            dtype=torch.long,
            device=self.device,
        )

        # Prefill (populates KV cache)
        self._mm._prefill(inputs_embeds, mask, pos_ids, None)

        # Create EncodedImage with cached KV state
        encoded = EncodedImage(
            pos=inputs_embeds.size(1),
            caches=[
                (
                    b.kv_cache.k_cache[
                        :, :, : inputs_embeds.size(1), :
                    ].clone(),
                    b.kv_cache.v_cache[
                        :, :, : inputs_embeds.size(1), :
                    ].clone(),
                )
                for b in self._mm.text.blocks
            ],
        )

        # Use model's query with our encoded image
        result = self.model.query(
            image=encoded,
            question="What do you see in this image?",
            stream=False,
        )
        return result["answer"].strip()

    @torch.no_grad()
    def classify(self, pil_image, source_text, target_text):
        """Classify image by description + keyword matching."""
        desc = self.describe(pil_image)
        desc_lower = desc.lower()

        source_kw = source_text.replace("a photo of ", "").strip()
        target_kw = target_text.replace("a photo of ", "").strip()

        has_source = source_kw.lower() in desc_lower
        has_target = target_kw.lower() in desc_lower

        if has_target and not has_source:
            prediction = target_text
            label = "B"
            target_selected = True
        elif has_source and not has_target:
            prediction = source_text
            label = "A"
            target_selected = False
        elif has_source and has_target:
            prediction = "both"
            label = "?"
            target_selected = False
        else:
            prediction = "neither"
            label = "?"
            target_selected = False

        return {
            "prediction": prediction,
            "prediction_label": label,
            "target_selected": target_selected,
            "raw_response": desc,
            "has_source": has_source,
            "has_target": has_target,
            "method": "keyword",
        }

    def tensor_to_pil(self, tensor):
        """Convert [1, 3, H, W] tensor in [0, 1] to PIL Image."""
        tensor = tensor[0].detach().cpu().clamp(0, 1)
        arr = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
        return Image.fromarray(arr)
