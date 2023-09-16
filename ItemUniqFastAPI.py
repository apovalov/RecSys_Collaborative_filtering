from typing import Tuple

import os
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every
from sklearn.neighbors import KernelDensity

DIVERSITY_THRESHOLD = 10

app = FastAPI()
embeddings = {}


@app.on_event("startup")
@repeat_every(seconds=10)
def load_embeddings() -> dict:
    """Load embeddings from file."""

    # Load new embeddings each 10 seconds
    path = os.path.join(os.path.dirname(__file__), "embeddings.npy")
    embeddings_raw = np.load(path, allow_pickle=True).item()
    for item_id, embedding in embeddings_raw.items():
        embeddings[item_id] = embedding

    return {}


@app.get("/uniqueness/")
def uniqueness(item_ids: str) -> dict:
    """Calculate uniqueness of each product"""

    # Parse item IDs
    item_ids = [int(item) for item in item_ids.split(",")]

    # Default answer
    item_uniqueness = {item_id: 0.0 for item_id in item_ids}

    # Calculate uniqueness
    item_embeddings = [embeddings[item_id] for item_id in item_ids]
    uniqueness_scores = kde_uniqueness(np.array(item_embeddings))

    for i, item_id in enumerate(item_ids):
        item_uniqueness[item_id] = uniqueness_scores[i]

    return item_uniqueness


@app.get("/diversity/")
def diversity(item_ids: str) -> dict:
    """Calculate diversity of group of products"""

    # Parse item IDs
    item_ids = [int(item) for item in item_ids.split(",")]

    # Calculate diversity
    item_embeddings = [embeddings[item_id] for item_id in item_ids]
    reject, group_diversity_score = group_diversity(np.array(item_embeddings), DIVERSITY_THRESHOLD)

    response = {"diversity": group_diversity_score, "reject": reject}
    return response



def kde_uniqueness(embeddings: np.ndarray) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on KDE.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    # Set bandwidth (you might need to adjust this based on your data)
    bandwidth = 1

    # Initialize the Kernel Density Estimation model
    kde = KernelDensity(bandwidth=bandwidth, metric='euclidean', kernel='gaussian')

    # Fit the KDE model to the embeddings
    kde.fit(embeddings)

    # Evaluate the KDE model at each embedding point
    kde_scores = kde.score_samples(embeddings)

    # Convert the scores to uniqueness estimates
    # We use exponential transformation to ensure positive values
    # (higher values indicate higher uniqueness)

    uniqueness_scores = 1 / np.exp(kde_scores)

    return uniqueness_scores


def group_diversity(embeddings: np.ndarray, threshold: float) -> Tuple[bool, float]:
    """Calculate group diversity based on kde uniqueness.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group
    threshold: float :
       group deversity threshold for reject group

    Returns
    -------
    Tuple[bool, float]
        reject
        group diverstity
    """
    # Calculate the uniqueness scores using the KDE uniqueness metric
    uniqueness_scores = kde_uniqueness(embeddings)

    # Calculate the group diversity as the mean uniqueness score
    group_diversity_score = np.mean(uniqueness_scores)

    # Check if the group diversity is below the threshold
    reject = group_diversity_score < threshold

    return reject, group_diversity_score


def main() -> None:
    """Run application"""
    uvicorn.run("main:app", host="localhost", port=5000)


if __name__ == "__main__":
    main()
