from ultralytics import SAM, FastSAM


class FastSAMPredictor():
    """
    FastSAM segmentation backend (extremely fast, promptable).
    Good for large-object segmentation and grounding.

    Supports:
        - Free-form prompt ("big objects", "furniture", etc.)
        - Retina masks
    """

    def __init__(self, cfg):
        ckpt = cfg.paths["fastsam_ckpt"]
        self.device = cfg.device
        print(f"[FastSAM] Loading {ckpt} → {self.device}")

        self.model = FastSAM(ckpt)

    def __call__(self, rgb_full, prompt=None):
        H, W = rgb_full.shape[:2]
        imgsz = max(H, W)

        results = self.model(
            rgb_full,
            texts=prompt,
            device=self.device,
            retina_masks=True,
            imgsz=imgsz,
            verbose=False,
        )

        if results[0].masks is None:
            return []
        return results[0].masks.data.cpu().numpy().astype(bool)


class MobileSAMPredictor():
    """
    Mobile-SAM (efficient SAM).
    Useful when SAM-like quality is needed but on-device speed matters.
    """

    def __init__(self, cfg):
        ckpt = cfg.paths.get("mobile_sam_ckpt")
        self.device = cfg.device
        if ckpt is None:
            raise ValueError("Missing `mobile_sam_ckpt` in cfg.paths.")

        print(f"[MobileSAM] Loading {ckpt} → {self.device}")

        self.model = SAM(ckpt)
        self.model.to(self.device)

    def __call__(self, rgb_full, bboxes, verbose=False, prompt=None):
        if prompt:
            raise NotImplementedError("MobileSAMPredictor does not support text prompt-based segmentation.")
        results = self.model.predict(rgb_full, bboxes=bboxes, device=self.device, verbose=verbose)
        if results[0].masks is None:
            return []
        return results[0].masks.data.cpu().numpy().astype(bool)


class SAM2Predictor():
    """
    SAM2 (Segment Anything 2).
    Highest accuracy among the SAM family, slowest runtime.

    Wrapped the same way as SAM/SAM2 from ultralytics.
    """

    def __init__(self, cfg):
        ckpt = cfg.paths.get("sam2_ckpt")
        self.device = cfg.device
        if ckpt is None:
            raise ValueError("Missing `sam2_ckpt` in cfg.paths.")

        print(f"[SAM2] Loading {ckpt} → {self.device}")

        self.model = SAM(ckpt)   # ultralytics SAM loads SAM2 models too
        self.model.to(self.device)

    def __call__(self, rgb_full, bboxes, verbose=False, prompt=None):
        if prompt:
            raise NotImplementedError("SAM2Predictor does not support text prompt-based segmentation.")
        results = self.model.predict(rgb_full, bboxes=bboxes, device=self.device, verbose=verbose)
        if results[0].masks is None:
            return []
        return results[0].masks.data.cpu().numpy().astype(bool)
