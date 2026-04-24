# Intro to NLP

Natural Language Processing (NLP) is how computers understand and generate text. In this repo, language models are used to caption objects and extract spatial relationships ("the cup is on top of the desk"). This page covers what you need to know.

---

## Large Language Models (LLMs)

An **LLM** is a neural network trained on massive text datasets to predict the next word (or token) in a sequence. Through this simple objective, it learns grammar, reasoning, facts, and how to follow instructions.

```
LLM: next-token prediction

  Input:  "The cup is on top of the"
  Output: "desk"  (most likely next token)

  Chain many predictions together → full sentence generation
```

### Tokens

LLMs do not process characters or words — they process **tokens** (roughly word pieces):

```
"the laptop is on the desk"
  →  ["the", " laptop", " is", " on", " the", " desk"]
  →  [  264,    21132,    374,    389,    264,   15051]  (token IDs)
```

### How This Repo Uses LLMs

After building the 3D map, for each pair of nearby objects, an LLM (Llama-4 via Groq API or GPT-4o via OpenAI) is given image crops and asked:

```
Prompt:
  "Object A is a laptop at position [1.2, 1.8, 0.82].
   Object B is a desk  at position [1.2, 2.0, 0.75].
   What is the spatial relation from A to B?
   Choose one: on_top_of / left_of / near / in_front_of / ..."

Response: "on_top_of"
```

The LLM does not need to be fine-tuned. It reasons from its pre-training knowledge.

---

## Encoders

A **text encoder** converts a string into a fixed-length embedding vector — the same idea as image embeddings, but for text.

```
Text encoder:

  "a black office chair"  ──► encoder ──► [0.21, −0.44, 0.07, …]
                                                  ↑
                                          512-D vector

  "a wooden desk"         ──► encoder ──► [0.05,  0.31, −0.12, …]
```

SigLIP has both an image encoder and a text encoder that map into the **same** vector space — so you can compare an image directly to a text description using cosine similarity.

---

## Cosine Similarity

The standard way to compare two embedding vectors is **cosine similarity**:

\[
\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}
\]

It measures the **angle** between two vectors, regardless of their magnitude:

```
sim = +1.0  →  same direction  (identical meaning)
sim =  0.0  →  perpendicular   (unrelated)
sim = −1.0  →  opposite        (opposite meaning)
```

```python
import numpy as np

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Between all pairs of N embeddings at once (vectorized)
# a: (N, D)  — N query vectors
# b: (M, D)  — M database vectors
a_n = a / np.linalg.norm(a, axis=1, keepdims=True)
b_n = b / np.linalg.norm(b, axis=1, keepdims=True)
sim_matrix = a_n @ b_n.T    # (N, M)
```

This is how **object tracking** works: a new detection's SigLIP embedding is compared to all tracked objects. The one with the highest cosine similarity is the match.

---

## Vector Search (Nearest Neighbor)

Once every item in a database has an embedding, finding the most similar item to a query is just: *which vector in the database is closest to my query vector?*

```
Database embeddings (one per reference image):

  Frame 0   ────────────────────────────────► ·
  Frame 1   ──────────────────────►·
  Frame 2   ────────────────────────────────────►·
  ...

  Query     ──────────────────────────────────► ·
               ↑
        cosine similarity computed to every frame

  Best match = Frame 0  (smallest angle)
  → Robot is near Frame 0's recorded position
```

This is the core of **Visual Place Recognition** in the localization module.

---

## VLAD: Aggregating Many Vectors Into One

DINOv2 produces 196 patch embeddings per image (one per 16×16 patch). To get a **single descriptor** for retrieval, VLAD aggregates them:

1. **K-Means** first — cluster all patch embeddings from the database into \(K\) cluster centers \(\mathbf{c}_1, \ldots, \mathbf{c}_K\)
2. For each new image, assign each patch to its nearest cluster and sum the residuals:

\[
\mathbf{V}_k = \sum_{i \;:\; \text{NN}(\mathbf{f}_i) = k} (\mathbf{f}_i - \mathbf{c}_k)
\]

3. Concatenate all \(K\) blocks and L2-normalize the result.

```
VLAD construction:

  196 patch embeddings  →  assign each to nearest cluster
                        →  sum residuals per cluster
                        →  [V₀ | V₁ | … | V_{K-1}]  L2-normalized
                              ↑ one compact descriptor per image
```

A single VLAD descriptor captures *how* this image's patches deviate from the average appearance of each visual pattern — far richer than a single CLS token.

---

## Summary: NLP Components in the Repo

| Component | Model | Where |
|-----------|-------|-------|
| Relation extraction | Llama-4 (Groq) or GPT-4o | `models/llm.py` |
| Image + text embedding | SigLIP | `models/embedding.py` |
| Object captioning | BLIP-2 | `models/captioning.py` |
| VLAD aggregation | (math, no model) | `core/jax_helper.py` |
| Cosine similarity | (math, no model) | `core/jax_helper.py` |
