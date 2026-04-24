import pandas as pd
import numpy as np

from spot_semantic_mapping.core.jax_helper import cdist, cosine_similarity_jax
from spot_semantic_mapping.localization.encoder import ImageEncoder


def prepare_embeddings(vision_transformer, images, query_frames, methods, ts, grayscale=False, cropping=False):
    res = {k: {} for k in methods}

    if not isinstance(query_frames, np.ndarray):
        query_frames = np.array(query_frames)

    if cropping:
        h, w = query_frames.shape[1:3]
        crop_h, crop_w = int(h * 0.75), int(w * 0.75)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        query_frames = query_frames[:, start_h:start_h+crop_h, start_w:start_w+crop_w, :]

    if grayscale:
        query_frames = np.stack([query_frames.mean(-1)] * 3, axis=-1)

    for method, cfg in methods.items():
        encoder = ImageEncoder(vision_transformer)
        embs = encoder.embed(images, cfg["patches"], cfg["agg_method"], cfg["num_clusters"], grayscale=cfg["grayscale"], load=False)

        print(f"Embedding queries for method: {method}")
        X = encoder.embed(query_frames, patches=cfg["patches"], agg_method=cfg["agg_method"], num_clusters=cfg['num_clusters'], grayscale=cfg['grayscale'], save=False, load=False)

        res[method] = {
            "embeddings": embs,
            "encoder": encoder,
            "X": X,
            "ts": ts
        }

    return res


def localize(X_emb, db_emb):
    """
    Match query embeddings against a database.

    Returns:
        im_scores   (N_query, N_db) cosine similarity matrix
        sorted_ind  (N_query, N_db) indices sorted by descending score
    """
    im_scores = cosine_similarity_jax(X_emb, db_emb)
    sorted_ind = np.argsort(-im_scores)
    return im_scores, sorted_ind


def localize_at_t(encoded_data, ground_truth, method, t, n_views=1):
    correct_db_indices = ground_truth[t][1]

    X = encoded_data[method]["X"][encoded_data[method]["ts"] == t][:n_views]
    cosine_similarity_scores = np.array(cosine_similarity_jax(X, encoded_data[method]["embeddings"]))
    image_scores = np.sum(cosine_similarity_scores, axis=0)
    predictions = np.argsort(image_scores)[0]

    return image_scores, correct_db_indices, predictions


def retrieve_subgraphs(dataset, ordered_indices, g, top_k=5, window=5):
    """
    Retrieve the scene subgraph around the top-k localized positions.

    Parameters
    ----------
    dataset : dict
        Must contain ``db_images`` and ``db_traj`` arrays.
    ordered_indices : np.ndarray
        Image indices ranked by localization score (descending).
    g : dict
        Scene graph with ``nodes`` and ``edges`` lists.
    top_k : int
        Number of top retrieved images to consider.
    window : float
        Spatial window radius (meters) for node inclusion.

    Returns
    -------
    dict with ``nodes`` (DataFrame) and ``edges`` (DataFrame).
    """
    nodes = pd.DataFrame(g['nodes'])
    edges = pd.DataFrame(g['edges'])
    n_pos = np.array(list(nodes['position']))
    retrieved_images_pos = np.array(dataset['db_traj'][ordered_indices][:top_k])  # (k, 3)

    dist_matrix = cdist(retrieved_images_pos, n_pos)
    node_mask = np.array(np.any(dist_matrix <= window, axis=0), dtype=bool).flatten()

    sub_nodes = nodes[node_mask]

    if sub_nodes.empty:
        print("State: No objects found within the specified window.")
        return {'nodes': [], 'edges': []}

    valid_oids = set(sub_nodes['oid'])
    sub_edges = edges[edges['src_id'].isin(valid_oids) & edges['dst_id'].isin(valid_oids)]

    return {"nodes": sub_nodes, "edges": sub_edges}


def print_subgraph(subgraph):
    """Return a formatted string summary of a retrieved subgraph."""
    sub_nodes, sub_edges = subgraph['nodes'], subgraph['edges']

    if sub_nodes.empty:
        return "="*40 + "\nSUBGRAPH STATE SUMMARY\n" + "="*40 + "\n  No objects detected within the current window.\n" + "="*40 + "\n"

    room_counts = sub_nodes['room'].value_counts()
    likely_room = room_counts.idxmax()

    summary = []
    summary.append("\n" + "="*40)
    summary.append("SUBGRAPH STATE SUMMARY")
    summary.append("="*40)
    summary.append(f"Likely Current Room: {likely_room}\n")

    summary.append("Objects per room (within window):")
    for room, count in room_counts.items():
        summary.append(f"  - {room}: {count} objects")

    summary.append(f"\nObjects detected in {likely_room}:")
    room_objects = sub_nodes[sub_nodes['room'] == likely_room]['class_name'].tolist()
    summary.append(f"  {', '.join(room_objects)}")

    summary.append("\nObject Relations (Induced Subgraph):")
    if not sub_edges.empty:
        for _, row in sub_edges.iterrows():
            summary.append(f"  - [{row['src_name']}] {row['relation']} [{row['dst_name']}]")
    else:
        summary.append("  - No local relations found between these objects.")

    summary.append("="*40 + "\n")
    return "\n".join(summary)
