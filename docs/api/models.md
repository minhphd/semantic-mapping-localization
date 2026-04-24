# Models

Model wrappers for all vision and language models used in the pipeline.

---

## Detection · `models.detection`

::: spot_semantic_mapping.models.detection
    options:
      members:
        - YOLODetector

---

## Segmentation · `models.segmentation`

::: spot_semantic_mapping.models.segmentation
    options:
      members:
        - SAM2Predictor
        - MobileSAMPredictor
        - FastSAMPredictor

---

## Captioning · `models.captioning`

::: spot_semantic_mapping.models.captioning
    options:
      members:
        - BlipCaptioner

---

## Embeddings · `models.embedding`

::: spot_semantic_mapping.models.embedding
    options:
      members:
        - DinoModel
        - SiglipModel

---

## LLMs · `models.llm`

::: spot_semantic_mapping.models.llm
    options:
      members:
        - GroqModel
        - OpenaiModel
        - OpenaiEmbedding
