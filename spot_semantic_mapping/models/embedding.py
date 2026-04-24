import torch
from transformers import (
    AutoModel,
    AutoImageProcessor,
    AutoProcessor,
)
import numpy as np


class CustomEncoder:
    """Base class for vision + text embedding."""

    def embed_images(self, crops_pil):
        raise NotImplementedError

    def embed_texts(self, texts):
        raise NotImplementedError


class DinoModel:
    """
    DINO / DINOv2 image encoder (image-only).
    Provides:
        - embed_images() → vision embeddings (CLS token)
        - embed_images_by_patch() → patch-level embeddings for VLAD

    Notes:
        - No text tower (embed_texts not supported).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        ckpt = cfg.paths["dino_ckpt"]  # e.g. "facebook/dinov2-base"
        print(f"[DINO] Loading {ckpt} → {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(ckpt, use_fast=True)
        self.model = AutoModel.from_pretrained(
            ckpt,
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()

    def embed_images(self, crops_pil):
        if len(crops_pil) == 0:
            return None

        with torch.no_grad():
            inputs = self.processor(images=crops_pil, return_tensors="pt").to(self.device)
            out = self.model(**inputs)

            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                h = out.last_hidden_state  # [B, T, D]
                feats = h[:, 0, :] if h.shape[1] >= 1 else h.mean(dim=1)
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                feats = out.pooler_output
            else:
                raise RuntimeError("DINO model output does not contain last_hidden_state/pooler_output")

            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)

        return feats.cpu().numpy()

    def embed_images_by_patch(self, crops_pil):
        if len(crops_pil) == 0:
            return None

        patch_size = self.model.config.patch_size if hasattr(self.model.config, "patch_size") else 16
        with torch.no_grad():
            inputs = self.processor(images=crops_pil, return_tensors="pt").to(self.device)
            _, _, img_height, img_width = inputs.pixel_values.shape
            num_patches_height = img_height // patch_size
            num_patches_width = img_width // patch_size

            last_hidden_states = self.model(**inputs)[0]
            patch_features = last_hidden_states[:, 1:, :].unflatten(
                1, (num_patches_height, num_patches_width)
            )

        patch_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-12)
        return patch_features.cpu().numpy()

    def embed_texts(self, texts):
        raise NotImplementedError("DINO is image-only; no text embeddings.")

    def classify_caption(self, caption):
        raise NotImplementedError("DINO is image-only; no caption classification.")


class SiglipModel():
    """
    SigLIP image + text encoder (open-set).
    Provides:
        - embed_images() → vision embeddings
        - embed_texts() → text embeddings
        - classify_caption() → open-set text classification
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        ckpt = cfg.paths["siglip_ckpt"]
        print(f"[SigLIP] Loading {ckpt} → {self.device}")

        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt).to(self.device)
        self.model.eval()

        self.class_names = cfg.landmarks["classes"]
        self.class_embs = self._build_text_embeddings()

    def embed_images(self, crops_pil):
        if len(crops_pil) == 0:
            return None

        with torch.no_grad():
            inputs = self.processor(images=crops_pil, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.cpu().numpy()

    def embed_texts(self, texts):
        if len(texts) == 0:
            return None

        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.cpu().numpy()

    def _build_text_embeddings(self):
        with torch.no_grad():
            text_inputs = self.processor(
                text=self.class_names,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            feats = self.model.get_text_features(**text_inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.cpu().numpy()

    def classify_caption(self, caption):
        with torch.no_grad():
            inputs = self.processor(
                text=[caption],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            emb = self.model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb_np = emb.cpu().numpy().reshape(-1)

        sims = emb_np @ self.class_embs.T
        return self.class_names[int(sims.argmax())]
