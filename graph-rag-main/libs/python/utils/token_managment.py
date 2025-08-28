import hashlib
from abc import abstractmethod
from typing import List

import tiktoken
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from libs.python.schemas.basic_models import LLMModel


def hash_string(
    input_string,
    algorithm="md5",
):
    """
    Hashes an input string using the specified algorithm.

    :param input_string: The string to be hashed.
    :param algorithm: The hashing algorithm to use (default is 'sha256').
    :return: The hexadecimal hash of the input string.
    """
    # Create a new hash object using the specified algorithm
    hash_object = hashlib.new(algorithm)

    # Update the hash object with the bytes of the input string
    hash_object.update(input_string.encode())

    # Return the hexadecimal representation of the hash
    return hash_object.hexdigest()


class TokenManager:
    encodings = {
        "o200k_base": tiktoken.get_encoding("o200k_base"),
        "cl100k_base": tiktoken.get_encoding("cl100k_base"),
    }

    @classmethod
    def num_tokens_from_string(
        cls, string: str, model_name: LLMModel | str = LLMModel.GPT_4O_MINI
    ) -> int:
        """Returns the number of tokens in a text string."""

        model_name = str(model_name)

        num_tokens = len(
            cls.encodings[tiktoken.encoding_name_for_model(model_name)].encode(string)
        )
        return num_tokens

    @classmethod
    def encode(
        cls, string: str, model_name: LLMModel = LLMModel.GPT_4O_MINI
    ) -> List[int]:
        return cls.encodings[tiktoken.encoding_name_for_model(model_name.value)].encode(
            string
        )

    @classmethod
    def decode(
        cls, tokens: List[int], model_name: LLMModel = LLMModel.GPT_4O_MINI
    ) -> str:
        return cls.encodings[tiktoken.encoding_name_for_model(model_name.value)].decode(
            tokens
        )


class TextPreProcessorBase:
    @abstractmethod
    def create_chunk_dataset(
        cls,
        content: str,
        src: str,
        llm_model: LLMModel,
    ) -> List[Document]:
        pass

    @abstractmethod
    def process(cls, content: str, src: str, chunk_size: int) -> List[Document]:
        """Process Text Provide src (e.g. url, link to article)"""
        pass


class TextPreProcessor(TextPreProcessorBase):
    @classmethod
    def create_chunk_dataset(
        cls,
        content: str,
        src: str,
        chunk_size: int,
    ) -> List[Document]:
        h = hash_string(content)
        uid = h[:12]
        chunk_size = chunk_size
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=40,
            length_function=lambda x: TokenManager.num_tokens_from_string(
                x, model_name="text-embedding-ada-002"
            ),
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_text(content)
        docs = []
        for i, chunk in enumerate(chunks):
            lc_doc = Document(
                page_content=chunk, metadata={"id": f"{uid}-{i}", "src": src}
            )
            docs.append(lc_doc)
        return docs

    @classmethod
    def process(cls, content: str, src: str, chunk_size: int) -> List[Document]:
        """Process Text Provide src (e.g. url, link to article)"""
        dcs = cls.create_chunk_dataset(content=content, src=src, chunk_size=chunk_size)
        return dcs
