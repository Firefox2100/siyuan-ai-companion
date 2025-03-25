import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from siyuan_ai_companion.consts import QDRANT_COLLECTION_NAME


class RagDriver:
    transformer: SentenceTransformer = None
    client: QdrantClient = None

    def __init__(self):
        if RagDriver.transformer is None:
            RagDriver.transformer = SentenceTransformer('all-MiniLM-L6-v2')

        if RagDriver.client is None:
            RagDriver.client = QdrantClient()

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

    def add_note(self,
                 note_id: str,
                 note_content: str,
                 ):
        vector = self.transformer.encode(
            sentences=note_content,
            normalize_embeddings=True,
        ).tolist()

        point = PointStruct(
            id=self._hash_id(note_id),
            vector=vector,
            payload={
                'noteId': note_id,
                'noteContent': note_content,
            }
        )

        self.client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[point],
        )

    def update_note(self,
                    note_id: str,
                    note_content: str,
                    ):
        self.add_note(
            note_id=note_id,
            note_content=note_content
        )

    def delete_note(self,
                    note_id: str,
                    ):
        self.client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector={'points': [self._hash_id(note_id)]},
        )

    def delete_all(self):
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
                'noteId': hit.payload['noteId'],
                'noteContent': hit.payload['noteContent'],
                'score': hit.score,
            })

        return results
