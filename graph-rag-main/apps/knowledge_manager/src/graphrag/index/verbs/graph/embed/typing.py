"""A module containing different lists and dictionaries."""

# Use this for now instead of a wrapper
from typing import Any

NodeList = list[str]
EmbeddingList = list[Any]
NodeEmbeddings = dict[str, list[float]]
"""Label -> Embedding"""
