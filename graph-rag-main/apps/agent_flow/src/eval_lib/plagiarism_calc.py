import os
import re
import time
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Union, Tuple, Any

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from libs.python.utils.request_utils import asend_request

ENDPOINT_GET_SOURCE_CONTENT = os.getenv(
    "ENDPOINT_GET_SOURCE_CONTENT",
    "http://localhost:8099/data/get_content_source_by_urls",
)


# Function to find the sublist index for a given merged list index
async def find_sublist_index(merged_index: int, sublist_lengths: List[int]) -> int:
    cumulative_length = 0
    for i, length in enumerate(sublist_lengths):
        cumulative_length += length
        if merged_index < cumulative_length:
            return i
    return -1  # If the index is out of bounds (shouldn't happen with valid input)


async def preprocess_text(text: str) -> str:
    # Remove unwanted symbols
    text = re.sub(r"[^\w\s]", "", text)
    # Remove words longer than 20 characters
    text = " ".join([word for word in text.split() if len(word) <= 20])
    return text


async def split_words_into_chunks(
    words: List[str], chunk_size: int, overlap_size: int
) -> List[str]:
    """
    Splits a list of words into chunks of specified size with overlapping windows.

    Args:
    words (list): The list of words to split.
    chunk_size (int): The size of each chunk.
    overlap_size (int): The size of the overlapping window between chunks.

    Returns:
    list: A list of strings where each string is a chunk with overlapping words.
    """
    if chunk_size <= overlap_size:
        raise ValueError("chunk_size must be greater than overlap_size")

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]

        # Calculate the start and end of the overlap for the next chunk
        overlap_start = max(0, end - overlap_size)
        next_chunk_start = overlap_start

        chunks.append(" ".join(chunk))

        # Move to the start of the next chunk
        start = next_chunk_start

    return chunks


async def sliding_chunk_text(
    text: str, chunk_size: int, overlap_size: int = 4
) -> List[str]:

    words = nltk.word_tokenize(text)
    chunks = await split_words_into_chunks(
        words=words, chunk_size=chunk_size, overlap_size=overlap_size
    )
    return chunks


def create_hypervector(dimensions: int = 10000) -> np.ndarray:
    # Function to create random hypervector
    return np.random.choice([-1, 1], size=(dimensions,))


async def hypervector_similarity(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    # Function to compute similarity between hypervectors
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


async def encode_text_to_hypervector(
    text: str, word_hypervectors, dimensions: int = 10000
) -> np.ndarray:
    # Function to encode text chunk into a hypervector
    words = text.split()
    hypervector = np.zeros(dimensions)
    for word in words:
        if word not in word_hypervectors:
            word_hypervectors[word] = create_hypervector(dimensions)
        word_hv = word_hypervectors[word]
        hypervector += word_hv
    return hypervector


async def top_k_max_indices(scores: np.ndarray, k: int) -> List[Tuple[int, int]]:
    if k > scores.size:
        raise ValueError("k cannot be larger than the number of scores")

    # Flatten the scores array and get the indices of the top k maximum scores
    flat_indices = np.argpartition(scores.flatten(), -k)[-k:]

    # Convert flat indices back to 2D indices
    top_2d_indices = np.unravel_index(flat_indices, scores.shape)

    # Get the pairs of indices
    top_pairs = list(zip(top_2d_indices[0], top_2d_indices[1]))

    # Sort these pairs by their corresponding score values in descending order
    top_pairs_sorted = sorted(
        top_pairs, key=lambda pair: scores[pair[0], pair[1]], reverse=True
    )

    return top_pairs_sorted


async def get_statistics(
    scores: np.ndarray,
    top_k: int,
    agent_chunks: List[str],
    all_chunks: List[str],
    len_slice_chunk: List[int],
    sources_content: List[Dict[str, Union[str, List[str]]]],
) -> Tuple[float, Dict[str, Any]]:
    top_k_scores_max_idx = await top_k_max_indices(scores, k=top_k)

    scores_flatten = scores.reshape(-1)
    mean = np.mean(scores_flatten)
    score = np.max(scores_flatten)

    quartiles = np.percentile(scores_flatten, [25, 50, 75])
    whiskers = [
        np.min(
            scores_flatten[
                scores_flatten >= (quartiles[0] - 1.5 * (quartiles[2] - quartiles[0]))
            ]
        ),
        np.max(
            scores_flatten[
                scores_flatten <= (quartiles[2] + 1.5 * (quartiles[2] - quartiles[0]))
            ]
        ),
    ]
    outliers = scores_flatten[
        (scores_flatten < whiskers[0]) | (scores_flatten > whiskers[1])
    ]

    top_k_scores = []
    for triplet_agent_idx, triplet_source_idx in top_k_scores_max_idx:
        idx_sublist = await find_sublist_index(
            triplet_source_idx, sublist_lengths=len_slice_chunk
        )
        top_k_scores.append(
            {
                "score": float(scores[triplet_agent_idx, triplet_source_idx]),
                "agent_chunk": agent_chunks[triplet_agent_idx],
                "source_chunk": all_chunks[triplet_source_idx],
                "source_content_index": idx_sublist,
                "source_name": sources_content[idx_sublist].get("Source"),
            }
        )

    statistics_data = {
        "mean": float(mean),
        "median": float(quartiles[1]),
        "q1": float(quartiles[0]),
        "q3": float(quartiles[2]),
        "whisker_low": float(whiskers[0]),
        "whisker_high": float(whiskers[1]),
        "outliers": [float(i) for i in outliers],
        "top_k_scores": top_k_scores,
    }
    return float(score), statistics_data


async def calculate_similarity(
    agent_answer: str,
    sources_content: List[Dict[str, Union[str, List[str]]]],
    chunk_size: int = 10,
    overlap_size: int = 4,
    dimensions: int = 10000,
    top_k: int = 500,
):
    # Preprocess both texts
    # Chunk both agent_answer and source_content with sliding window

    agent_answer = await preprocess_text(agent_answer)
    agent_chunks = await sliding_chunk_text(
        agent_answer, chunk_size=chunk_size, overlap_size=overlap_size
    )
    entire_text = ""
    for source_content in sources_content:
        source_content_content = source_content.get("Content")
        if isinstance(source_content_content, str):
            if len(source_content_content) != 0:
                entire_text += source_content_content
                source_content_preprocessed = await preprocess_text(
                    source_content_content
                )
                source_chunks = await sliding_chunk_text(
                    source_content_preprocessed,
                    chunk_size=chunk_size,
                    overlap_size=overlap_size,
                )
                source_content["Chunks"] = source_chunks

    # Combine agent chunks with source chunks
    set_chunks = [source_content["Chunks"] for source_content in sources_content]
    if len(set_chunks) != 0:
        len_slice_chunk = [len(sc) for sc in set_chunks]
        all_chunks = []
        for chunks_ in set_chunks:
            all_chunks.extend(chunks_)

        texts = agent_chunks + all_chunks

        # Calculate TF-IDF vectors
        vectorizer = TfidfVectorizer().fit_transform(texts)
        vectors = vectorizer.toarray()

        # Compute cosine similarity between each agent chunk and all source chunks
        cosine_similarities = cosine_similarity(
            vectors[: len(agent_chunks)], vectors[len(agent_chunks) :]
        )

        # Compute high-dimensional computing similarity
        word_hypervectors = defaultdict(lambda: create_hypervector(dimensions))
        hdc_similarities = np.zeros_like(cosine_similarities)
        agent_hypervectors = [
            await encode_text_to_hypervector(chunk, word_hypervectors, dimensions)
            for chunk in agent_chunks
        ]
        source_hypervectors = [
            await encode_text_to_hypervector(chunk, word_hypervectors, dimensions)
            for chunk in all_chunks
        ]

        for i, agent_hv in enumerate(agent_hypervectors):
            for j, source_hv in enumerate(source_hypervectors):
                hdc_similarities[i, j] = await hypervector_similarity(
                    agent_hv, source_hv
                )

        # Calculate HDC similarity for the entire text
        entire_agent_hv = await encode_text_to_hypervector(
            await preprocess_text(agent_answer), word_hypervectors
        )
        entire_source_hv = await encode_text_to_hypervector(
            await preprocess_text(entire_text), word_hypervectors
        )
        entire_hdc_similarity = await hypervector_similarity(
            entire_agent_hv, entire_source_hv
        )

        # Calculate Cosine similarity for the entire text
        vectorizer = TfidfVectorizer().fit_transform([agent_answer, entire_text])
        vectors = vectorizer.toarray()
        entire_cosine_similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

        tf_id_score, statistics_data_tf_idf = await get_statistics(
            scores=cosine_similarities,
            top_k=top_k,
            agent_chunks=agent_chunks,
            all_chunks=all_chunks,
            len_slice_chunk=len_slice_chunk,
            sources_content=sources_content,
        )
        hdc_score, statistics_data_hdc = await get_statistics(
            scores=hdc_similarities,
            top_k=top_k,
            agent_chunks=agent_chunks,
            all_chunks=all_chunks,
            len_slice_chunk=len_slice_chunk,
            sources_content=sources_content,
        )

        return tf_id_score, {
            "statistics_data_tf_idf": statistics_data_tf_idf,
            "statistics_data_hdc": statistics_data_hdc,
            "hdc_score": hdc_score,
            "entire_hdc_similarity": entire_hdc_similarity,
            "entire_tf_idf_cosine_similarity": entire_cosine_similarity,
        }
    else:
        return None, dict()


async def plagiarism_calc(
    response: str,
    content_sources: Dict[str, Dict[str, str]],
    chunk_size: int = 10,
    overlap_size: int = 4,
    dimensions: int = 10000,
):
    st_time = time.time()
    list_of_urls_of_sources = list(
        set(
            (
                content_source.get("Source")
                for content_source in content_sources.values()
            )
        )
    )  # get uniq list of sources

    response_source_contents_ = await asend_request(
        data={"urls": list_of_urls_of_sources}, endpoint=ENDPOINT_GET_SOURCE_CONTENT
    )
    source_retrival_info = list()
    if response_source_contents_:
        response_source_contents = response_source_contents_.json()
        for response_source_content in response_source_contents.get("results"):
            url_source = response_source_content.get("url")
            content_source = response_source_content.get("content")

            source_retrival_info.append(
                {"Content": content_source, "Source": url_source}
            )

    score, additional_metrics = await calculate_similarity(
        agent_answer=response,
        sources_content=source_retrival_info,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        dimensions=dimensions,
    )

    process_metrics = [
        {
            "additional_metrics": {},
            "time_exe": time.time() - st_time,
            "name": "not_llm",
            "type_process": "llm_process",
            "input_metrics": {"input_cost": -1, "input_tokens": -1},
            "output_metrics": {"output_cost": -1, "output_tokens": -1},
            "total_metrics": {"total_cost": -1, "total_tokens": -1},
            "count_input_tokens_exceeded": -1,
        }
    ]
    return score, additional_metrics, process_metrics
