import hashlib
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from siyuan_ai_companion.consts import QDRANT_COLLECTION_NAME, QDRANT_LOCATION
from .siyuan_api import SiyuanApi


class RagDriver:
    transformer: SentenceTransformer = None
    client: QdrantClient = None

    def __init__(self):
        if RagDriver.transformer is None:
            RagDriver.transformer = SentenceTransformer('all-MiniLM-L6-v2')

        if RagDriver.client is None:
            RagDriver.client = QdrantClient(
                location=QDRANT_LOCATION,
            )

        if not RagDriver.client.collection_exists(QDRANT_COLLECTION_NAME):
            RagDriver.client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=RagDriver.transformer.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE,
                )
            )

    @staticmethod
    def _hash_id(note_id: str) -> int:
        return int.from_bytes(hashlib.md5(note_id.encode()).digest()[:8], "big")

    def add_block(self,
                  block_id: str,
                  block_content: str,
                  ):
        """
        Add a block to the vector index

        :param block_id: The ID of the block, used in SiYuan
        :param block_content: The content of the block, plain text
                              with markdown stripped
        :return: None
        """
        vector = self.transformer.encode(
            sentences=block_content,
            normalize_embeddings=True,
        ).tolist()

        point = PointStruct(
            id=self._hash_id(block_id),
            vector=vector,
            payload={
                'blockId': block_id,
            }
        )

        self.client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[point],
        )

    def add_blocks(self,
                         blocks: list[tuple[str, str]],
                         ):
        """
        Add multiple blocks to the vector index

        :param blocks: A list of tuples, each containing the ID
                       of the block and its content
        """
        points = []

        for block_id, block_content in blocks:
            vector = self.transformer.encode(
                sentences=block_content,
                normalize_embeddings=True,
            ).tolist()

            point = PointStruct(
                id=self._hash_id(block_id),
                vector=vector,
                payload={
                    'blockId': block_id,
                }
            )

            points.append(point)

        self.client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points,
        )

    def update_block(self,
                     block_id: str,
                     block_content: str,
                     ):
        """
        Update a block in the vector index

        For now this uses the same upsert method as add_block,
        which will replace the existing index
        :param block_id: The ID of the block, used in SiYuan
        :param block_content: The content of the block, plain text
                              with markdown stripped
        :return: None
        """
        self.add_block(
            block_id=block_id,
            block_content=block_content
        )

    def update_blocks(self,
                      blocks: list[tuple[str, str]],
                      ):
        """
        Update multiple blocks in the vector index

        For now this uses the same upsert method as add_block,
        which will replace the existing index
        :param blocks: A list of tuples, each containing the ID
                       of the block and its content
        :return: None
        """
        self.add_blocks(
            blocks=blocks,
        )

    def delete_block(self,
                     block_id: str,
                     ):
        """
        Delete a block from the vector index

        :param block_id: The ID of the block, used in SiYuan
        :return: None
        """
        self.client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector={'points': [self._hash_id(block_id)]},
        )

    def delete_all(self):
        """
        Delete all blocks from the vector index
        """
        self.client.delete_collection(
            collection_name=QDRANT_COLLECTION_NAME,
        )

        self.client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=self.transformer.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            )
        )

    def search(self,
               query: str,
               limit: int = 5,
               ) -> list[dict]:
        """
        Search for the most relevant blocks in the vector index

        :param query: The user message, in plain text
        :param limit: The number of results to return
        :return: A list of the most relevant blocks, with their IDs
                 and scores
        """
        query_vector = self.transformer.encode(
            sentences=query,
            normalize_embeddings=True,
        ).tolist()

        hits = self.client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=limit,
        )

        results = []

        for hit in hits.points:
            results.append({
                'blockId': hit.payload['blockId'],
                'score': hit.score,
            })

        return results

    async def build_prompt(self,
                           query: str,
                           limit: int = 5,
                           ) -> str:
        """
        Construct the prompt using the search results

        This prompt is used to generate the completion
        with the user message

        :param query: The user message
        :param limit: The number of search results to use
        :return: The prompt, added with the search results
        """
        search_results = self.search(
            query=query,
            limit=limit,
        )

        if not search_results:
            return query

        block_ids = set(
            result['blockId']
            for result in search_results
        )
        async with SiyuanApi() as siyuan:
            tasks = [
                siyuan.get_note_plaintext(note_id=block_id)
                for block_id in block_ids
            ]

            notes = await asyncio.gather(*tasks)

        prompt = 'Here are some documents that may help answer the question:\n\n'

        for note in notes:
            prompt += f'{note}\n\n'

        prompt += 'Answer the following question based on the documents above and your own knowledge:\n'
        prompt += query

        return prompt
