import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from spot_semantic_mapping.core.crops import concat_crops_horizontal


class BlipCaptioner():
    """
    BLIP-2 caption model (Flan-T5-XL).

    Designed to:
      - fuse multi-view crops horizontally
      - produce compact noun-phrase captions ("wooden table", "black sofa")
    """

    def __init__(self, cfg):
        print("[BLIP-2] Loading blip2-flan-t5-xl")

        self.device = cfg.device
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = Blip2Processor.from_pretrained(cfg.paths["blip_ckpt"])
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            cfg.paths["blip_ckpt"],
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def caption_single(self, crop_np, max_new_tokens=40):
        """
        Generate a caption for a single image crop.

        Parameters:
            crop_np: np.ndarray         (H, W, 3) RGB array
            max_new_tokens: int         maximum caption length

        Returns:
            caption string
        """
        if crop_np is None:
            return "(no caption)"

        pil_img = Image.fromarray(crop_np)

        prompt = (
            "Describe the central object in the image concisely in under 5 words. "
            "Ignore irrelevant background."
        )

        with torch.no_grad():
            inputs = self.processor(
                pil_img,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        return self.processor.decode(out[0], skip_special_tokens=True).strip()

    def caption_multi(self, crops_np, max_new_tokens=40):
        """
        Generate a caption for multi-view crops.

        Parameters:
            crops_np: List[np.ndarray]   list of (H, W, 3) RGB arrays
            max_new_tokens: int         maximum caption length

        Returns:
            caption string
        """
        fused = concat_crops_horizontal(crops_np)
        if fused is None:
            return "(no caption)"

        pil_img = Image.fromarray(fused)

        prompt = (
            "These images show different views of the same object. "
            "Describe the object concisely in under 5 words. "
            "Ignore irrelevant background."
        )

        with torch.no_grad():
            inputs = self.processor(
                pil_img,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        return self.processor.decode(out[0], skip_special_tokens=True).strip()
