import heapq
import numpy as np
from typing import Dict, List, Tuple
from itertools import combinations
from collections import defaultdict


class SimilarItems:
    """Similar items class"""

    @staticmethod
    def similarity(embeddings: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:
        """Calculate pairwise similarities between each item
        in embedding.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.

        Returns:
            Tuple[List[str], Dict[Tuple[int, int], float]]:
            List of all items + Pairwise similarities dict
            Keys are in form of (i, j) - combinations pairs of item_ids
            with i < j.
            Round each value to 8 decimal places.
        """
        result_dict = {}

        keys = list(embeddings)       
        combs = combinations(keys, 2)
 
        for item1_idx, item2_idx in combs:
            embedding_1 = embeddings[item1_idx]
            embedding_2 = embeddings[item2_idx]
            
            similarity = np.dot(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))
            result_dict[(item1_idx, item2_idx)] = round(similarity, 8)
        
        return result_dict

    @staticmethod
    def knn(
        sim: Dict[Tuple[int, int], float], top: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Return closest neighbors for each item.

        Args:
            sim (Dict[Tuple[int, int], float]): <similarity> method output.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, List[Tuple[int, float]]]: Dict with top closest neighbors
            for each item.
        """
        knn_dict = defaultdict(list)
        for (item_id_1, item_id_2), similarity in sim.items():
            heapq.heappush(knn_dict[item_id_1], (item_id_2, similarity))  # Fixed order of item_id and similarity
            heapq.heappush(knn_dict[item_id_2], (item_id_1, similarity))  # Fixed order of item_id and similarity
        knn_result = {}
        for item_id, neighbors in knn_dict.items():
            top_neighbors = heapq.nlargest(top, neighbors, key=lambda x: x[1])
            knn_result[item_id] = [(neighbor, similarity) for neighbor, similarity in top_neighbors]
        return knn_result

    @staticmethod
    def knn_price(
        knn_dict: Dict[int, List[Tuple[int, float]]],
        prices: Dict[int, float],
    ) -> Dict[int, float]:
        """Calculate weighted average prices for each item.
        Weights should be positive numbers in [0, 2] interval.

        Args:
            knn_dict (Dict[int, List[Tuple[int, float]]]): <knn> method output.
            prices (Dict[int, float]): Price dict for each item.

        Returns:
            Dict[int, float]: New prices dict, rounded to 2 decimal places.
        """
        knn_price_dict = {} 
        for item_id, neighbors in knn_dict.items():
            total_weight = 0
            weighted_price_sum = 0
            for neighbor, similarity in neighbors:
                weight = similarity + 1  # Calculate weight in [0, 2] interval
                total_weight += weight
                weighted_price_sum += prices[neighbor] * weight

            if total_weight > 0:
                knn_price_dict[item_id] = round(weighted_price_sum / total_weight, 2)
            else:
                knn_price_dict[item_id] = 0

        return knn_price_dict

    @staticmethod
    def transform(
        embeddings: Dict[int, np.ndarray],
        prices: Dict[int, float],
        top: int,
    ) -> Dict[int, float]:
        """Transforming input embeddings into a dictionary
        with weighted average prices for each item.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.
            prices (Dict[int, float]): Price dict for each item.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, float]: Dict with weighted average prices for each item.
        """
        knn_price_dict = {} 
        # Step 1: Calculate pairwise similarities
        sim = SimilarItems.similarity(embeddings)

        # Step 2: Find top closest neighbors for each item
        knn_dict = SimilarItems.knn(sim, top)

        # Step 3: Calculate weighted average prices for each item
        knn_price_dict = SimilarItems.knn_price(knn_dict, prices)

        return knn_price_dict

# embeddings = {
#     1: np.array([-1, -1, -1, 1]),
#     2: np.array([5, 5, 5, 4]),
#     3: np.array([10, 10, 10, 1]),
#     4: np.array([2, 2, 2, 2])
# }

# prices = {
#     1: 100.5,
#     2: 12.2, #0,49896 = 6,087312
#     3: 60.0,
#     4: 11.1  #0.50103 = 5,561433
# }

# # similarity = SimilarItems.similarity(embeddings=embeddings)
# # print('Similarity: ', similarity)
# # print('*********************')

# # sim = similarity.copy()
# # knn = SimilarItems.knn(sim=sim, top=3)

# # print('*********************')
# # print('KNN\n', knn)

# # print('*********************')
# # print('knn_price: ', SimilarItems.knn_price(knn, prices=prices))
