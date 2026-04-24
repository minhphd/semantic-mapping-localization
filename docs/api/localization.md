# Localization

The localization modules implement Visual Place Recognition using DINOv2 patch features and VLAD aggregation.

---

## `localization.localizer`

::: spot_semantic_mapping.localization.localizer
    options:
      members:
        - localize
        - retrieve_subgraphs
        - prepare_embeddings

---

## `localization.encoder`

::: spot_semantic_mapping.localization.encoder
    options:
      members:
        - ImageEncoder

---

## `localization.dataset`

::: spot_semantic_mapping.localization.dataset
    options:
      members:
        - build_database

---

## `localization.evaluation`

::: spot_semantic_mapping.localization.evaluation
    options:
      members:
        - compute_metrics_from_scores
