"""
RAG driver handles document embedding, vector storage,
retrival and prompt construction for the RAG model. It
also uses the SiyuanApi directly to get the full content
of the notes.
"""

import hashlib
import asyncio
import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from markdown_it import MarkdownIt

from siyuan_ai_companion.consts import APP_CONFIG, LOGGER
from .siyuan_api import SiyuanApi


class RagDriver:
    """
    The RAG driver for the vector index
    """
    transformer: SentenceTransformer = None
    client: QdrantClient = None
    _openai_tokenizer: tiktoken.Encoding | None = None
    _huggingface_tokenizer: PreTrainedTokenizerFast | None = None
    _selected_model: str = None

    def __init__(self):
        if self.transformer is None:
            RagDriver.transformer = SentenceTransformer('all-MiniLM-L6-v2')

        if self.client is None:
            RagDriver.client = QdrantClient(
                location=APP_CONFIG.qdrant_location,
            )

        if not self.client.collection_exists(APP_CONFIG.qdrant_collection_name):
            RagDriver.client.create_collection(
                collection_name=APP_CONFIG.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=RagDriver.transformer.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE,
                )
            )

    @property
    def selected_model(self) -> str:
        """
        The currently selected model for tokenisation
        """
        if self._selected_model is None:
            self.selected_model = 'gpt-3.5-turbo'

        return self._selected_model

    @selected_model.setter
    def selected_model(self, model_name: str):
        """
        Set the currently selected model for tokenisation

        :param model_name: The name of the model to use
        """
        if model_name == self._selected_model:
            # No need to set the model again
            return

        RagDriver._selected_model = model_name
        LOGGER.info('Selected model: %s', model_name)

        if model_name.startswith('gpt'):
            # OpenAI models
            RagDriver._openai_tokenizer = tiktoken.encoding_for_model(model_name)
            RagDriver._huggingface_tokenizer = None
        else:
            # Huggingface models
            RagDriver._huggingface_tokenizer = AutoTokenizer.from_pretrained(model_name)
            RagDriver._openai_tokenizer = None

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast | tiktoken.Encoding:
        """
        The currently selected tokeniser
        """
        if self.selected_model.startswith('gpt'):
            return self._openai_tokenizer

        return self._huggingface_tokenizer

    @staticmethod
    def _hash_id(note_id: str) -> int:
        return int.from_bytes(hashlib.md5(note_id.encode()).digest()[:8], 'big')

    def _estimate_tokens(self,
                         passage: str,
                         ) -> int:
        """
        Estimate the token usage of a passage based on a given model name

        :param passage: The passage to estimate the token usage for
        :return: The estimated number of tokens
        """
        return len(self.tokenizer.encode(passage))

    def _segment_document(self,
                          document: str,
                          matching_blocks: list[str],
                          max_tokens: int = 512,
                          current_level: int = None,
                          ) -> list[str]:
        """
        Segment the document based on Markdown structure and the matching block
        from the vector index.

        :param document: The document to segment, in Markdown format
        :param matching_blocks: The block from Qdrant, which was used to fetch the full
                                content of the note
        :param max_tokens: The maximum number of tokens that can be used by the segment
        :param current_level: (internal) the Markdown heading level currently used for splitting
        :return: The segmented document, in Markdown format
        """
        if not matching_blocks:
            raise ValueError('No matching blocks provided for segmentation')

        if self._estimate_tokens(passage=document) <= max_tokens:
            return [document]

        md = MarkdownIt()
        tokens = md.parse(document)

        # Extract heading levels
        all_levels = sorted(set(int(tok.tag[1]) for tok in tokens if tok.type == 'heading_open'))
        if not all_levels:
            # No headers found; fallback to paragraph split
            split = self._fallback_split(document, max_tokens)

            results = []
            for b in matching_blocks:
                for t in split:
                    if b in t:
                        results.append(t)

            return results

        split_level = current_level or all_levels[0]

        blocks = []
        current_title = None
        current_content = []

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == 'heading_open' and int(tok.tag[1]) == split_level:
                # Store previous block
                if current_title or current_content:
                    blocks.append(
                        (
                            current_title or '',
                            ''.join(getattr(t, 'content', '')
                                    for t in current_content).strip()
                        )
                    )
                    current_content = []

                current_title = tokens[i + 1].content if i + 1 < len(tokens) else ''
                i += 2  # skip heading_open and inline
            else:
                current_content.append(tok)
                i += 1

        if current_title or current_content:
            blocks.append(
                (
                    current_title or '',
                    ''.join(getattr(t, 'content', '')
                            for t in current_content).strip()
                )
            )

        # If only one block was produced, try a deeper header level
        if len(blocks) == 1:
            deeper_levels = [lvl for lvl in all_levels if lvl > split_level]
            if deeper_levels:
                return self._segment_document(
                    document,
                    matching_blocks,
                    max_tokens,
                    current_level=deeper_levels[0]
                )

            split = self._fallback_split(document, max_tokens)

            results = []
            for b in matching_blocks:
                for t in split:
                    if b in t:
                        results.append(t)

            return results

        # Match against relevant blocks
        results = []

        for b in matching_blocks:
            for title, text in blocks:
                if b in title or b in text:
                    if self._estimate_tokens(passage=text) > max_tokens:
                        results.extend(
                            self._segment_document(
                                text,
                                [b],
                                max_tokens,
                                current_level=split_level,
                            )
                        )
                    else:
                        results.append(text)

        LOGGER.debug('Segmented document: %s', results)
        return results

    def _fallback_split(self,
                        document: str,
                        max_tokens: int,
                        ) -> list[str]:
        """
        Split a document by paragraph if no headers are usable.

        This is a fallback method for splitting the document
        when no headers are found or the document is too long to be accepted.
        :param document: The document to split
        :param max_tokens: The maximum number of tokens that can be used by the segment
        """
        paragraphs = [p.strip() for p in document.split('\n\n') if p.strip()]
        segments = []
        current = ""

        for p in paragraphs:
            tentative = current + "\n\n" + p if current else p
            if self._estimate_tokens(tentative) <= max_tokens:
                current = tentative
            else:
                if current:
                    segments.append(current.strip())
                current = p

        if current:
            segments.append(current.strip())

        return segments

    def add_block(self,
                  block_id: str,
                  document_id: str,
                  block_content: str,
                  ):
        """
        Add a block to the vector index

        :param block_id: The ID of the block, used in SiYuan
        :param document_id: The ID of the document containing the block, used in SiYuan
        :param block_content: The content of the block, plain text
                              with Markdown stripped
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
                'documentId': document_id,
                'content': block_content,
            }
        )

        self.client.upsert(
            collection_name=APP_CONFIG.qdrant_collection_name,
            points=[point],
        )

        LOGGER.debug('Added block: %s', block_id)

    def add_blocks(self,
                   blocks: list[tuple[str, str, str]],
                   ):
        """
        Add multiple blocks to the vector index

        :param blocks: A list of tuples, each containing the ID
                       of the block, document ID and its content
        """
        points = []

        for block_id, block_content, document_id in blocks:
            vector = self.transformer.encode(
                sentences=block_content,
                normalize_embeddings=True,
            ).tolist()

            point = PointStruct(
                id=self._hash_id(block_id),
                vector=vector,
                payload={
                    'blockId': block_id,
                    'documentId': document_id,
                    'content': block_content,
                }
            )

            points.append(point)

        self.client.upsert(
            collection_name=APP_CONFIG.qdrant_collection_name,
            points=points,
        )

        LOGGER.debug('Added blocks: %s', [block[0] for block in blocks])

    def update_block(self,
                     block_id: str,
                     document_id: str,
                     block_content: str,
                     ):
        """
        Update a block in the vector index

        For now this uses the same upsert method as add_block,
        which will replace the existing index
        :param block_id: The ID of the block, used in SiYuan
        :param document_id: The ID of the document containing the block, used in SiYuan
        :param block_content: The content of the block, plain text
                              with Markdown stripped
        :return: None
        """
        self.add_block(
            block_id=block_id,
            document_id=document_id,
            block_content=block_content,
        )

        LOGGER.debug('Updated block: %s', block_id)

    def update_blocks(self,
                      blocks: list[tuple[str, str, str]],
                      ):
        """
        Update multiple blocks in the vector index

        For now, this uses the same upsert method as add_block,
        which will replace the existing index
        :param blocks: A list of tuples, each containing the ID
                       of the block, document ID and its content
        :return: None
        """
        self.add_blocks(
            blocks=blocks,
        )

        LOGGER.debug('Updated blocks: %s', [block[0] for block in blocks])

    def delete_block(self,
                     block_id: str,
                     ):
        """
        Delete a block from the vector index

        :param block_id: The ID of the block, used in SiYuan
        :return: None
        """
        self.client.delete(
            collection_name=APP_CONFIG.qdrant_collection_name,
            points_selector={'points': [self._hash_id(block_id)]},
        )

        LOGGER.debug('Deleted block: %s', block_id)

    def delete_all(self):
        """
        Delete all blocks from the vector index
        """
        self.client.delete_collection(
            collection_name=APP_CONFIG.qdrant_collection_name,
        )

        self.client.create_collection(
            collection_name=APP_CONFIG.qdrant_collection_name,
            vectors_config=VectorParams(
                size=self.transformer.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            )
        )

        LOGGER.debug('Deleted all blocks from the vector index')

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
        LOGGER.debug('Searching for blocks with query: %s', query)

        query_vector = self.transformer.encode(
            sentences=query,
            normalize_embeddings=True,
        ).tolist()

        try:
            hits = self.client.query_points(
                collection_name=APP_CONFIG.qdrant_collection_name,
                query=query_vector,
                limit=limit,
            )
        except ValueError:
            # No results found
            return []

        results = []
        LOGGER.debug('%s results found', len(hits.points))

        for hit in hits.points:
            results.append({
                'blockId': hit.payload['blockId'],
                'documentId': hit.payload['documentId'],
                'content': hit.payload['content'],
                'score': hit.score,
            })

        return results

    async def get_context(self,
                          query: str,
                          limit: int = 3,
                          max_tokens: int = 512,
                          ) -> list[str]:
        """
        Get the context for the given query

        :param query: The user message
        :param limit: The number of search results to use
        :param max_tokens: The maximum number of tokens that can be used by the segment
        :return: The context, in Markdown format
        """
        LOGGER.debug('Getting context for: %s', query)

        search_results = self.search(
            query=query,
            limit=limit,
        )

        if not search_results:
            return []

        document_ids = list(set(
            result['documentId']
            for result in search_results
        ))

        LOGGER.debug('Found %d documents', len(document_ids))

        async with SiyuanApi() as siyuan:
            tasks = [
                siyuan.get_note_markdown(note_id=document_id)
                for document_id in document_ids
            ]

            notes = await asyncio.gather(*tasks)

        # Segment the documents based on the matching blocks
        segments = []

        for i, document_id in enumerate(document_ids):
            matching_blocks = [
                r['content'] for r in search_results
                if r['documentId'] == document_id
            ]

            segments.extend(self._segment_document(
                document=notes[i],
                matching_blocks=matching_blocks,
                max_tokens=max_tokens,
            ))

        segments = list(set(segments))

        LOGGER.debug('Segments: %s', segments)

        return segments[:limit * 2]

    async def build_prompt(self,
                           query: str,
                           limit: int = 3,
                           max_tokens: int = 512,
                           ) -> str:
        """
        Construct the prompt using the search results

        This prompt is used to generate the completion
        with the user message

        :param query: The user message
        :param limit: The number of search results to use
        :param max_tokens: The maximum number of tokens that can be used by the segment
        :return: The prompt, added with the search results
        """
        contexts = await self.get_context(
            query=query,
            limit=limit,
            max_tokens=max_tokens,
        )

        prompt = 'Additional context:\n\n'
        prompt += '\n\n'.join(contexts)
        prompt += 'Question: ' + query
        prompt += '\n\nAnswer: \n\n'

        return prompt
