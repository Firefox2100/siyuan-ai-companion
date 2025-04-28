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
from markdown_it.token import Token

from siyuan_ai_companion.consts import APP_CONFIG, LOGGER
from siyuan_ai_companion.errors import RagDriverError
from .siyuan_api import SiyuanApi


class RagDriver:
    """
    The RAG driver for the vector index
    """
    transformer: SentenceTransformer = None
    client: QdrantClient = None
    _openai_tokenizer: tiktoken.Encoding | None = None
    _huggingface_tokenizer: PreTrainedTokenizerFast | None = None
    _selected_model: str | None = None
    _max_segment_tokens: int = 512

    def __init__(self):
        if RagDriver.transformer is None:
            LOGGER.info('Using transformer model: all-MiniLM-L6-v2')

            RagDriver.transformer = SentenceTransformer('all-MiniLM-L6-v2')

        if RagDriver.client is None:
            LOGGER.info('Connecting to Qdrant at %s', APP_CONFIG.qdrant_location)

            RagDriver.client = QdrantClient(
                location=APP_CONFIG.qdrant_location,
            )

        if not RagDriver.client.collection_exists(APP_CONFIG.qdrant_collection_name):
            LOGGER.info('Creating collection %s', APP_CONFIG.qdrant_collection_name)

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

            LOGGER.info('Using OpenAI tokenizer')
        else:
            # Huggingface models
            try:
                RagDriver._huggingface_tokenizer = AutoTokenizer.from_pretrained(model_name)

                LOGGER.info('Using Huggingface tokenizer')
            except (ValueError, OSError):
                # Model not recognized. Fallback to generic tokenizer
                RagDriver._huggingface_tokenizer = AutoTokenizer.from_pretrained(
                    'bert-base-uncased'
                )

                LOGGER.warning('Model not recognized. Using generic tokenizer')

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast | tiktoken.Encoding:
        """
        The currently selected tokeniser

        It may return two different classes, but they both implement
        the `.encode()` method, so can be used interchangeably
        """
        if self.selected_model.startswith('gpt'):
            LOGGER.debug('Selecting OpenAI tokenizer')
            return self._openai_tokenizer

        LOGGER.debug('Selecting Huggingface tokenizer')
        return self._huggingface_tokenizer

    @property
    def max_segment_tokens(self) -> int:
        return self._max_segment_tokens

    @max_segment_tokens.setter
    def max_segment_tokens(self, max_tokens: int):
        """
        Set the maximum number of tokens for a segment
        :param max_tokens: The maximum number of tokens
        """
        if max_tokens <= 0:
            raise RagDriverError('max_tokens must be greater than 0')

        RagDriver._max_segment_tokens = max_tokens
        LOGGER.info('Set max segment tokens to %d', max_tokens)

    @staticmethod
    def _hash_id(note_id: str) -> int:
        """
        Hash the note ID to a 64-bit integer
        :param note_id: The ID of the note, used in SiYuan
        :return: The hashed ID, as a 64-bit integer
        """
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
                          current_level: int = None) -> list[str]:
        """
        Segment the document based on Markdown structure and the matching block
        from the vector index.

        :param document: The document to segment, in Markdown format
        :param matching_blocks: The block from Qdrant, which was used to fetch the full
                                content of the note
        :param current_level: (internal) the Markdown heading level currently used for splitting
        :return: The segmented document, in Markdown format
        """
        if not matching_blocks:
            LOGGER.error('No matching blocks provided for segmentation')
            raise RagDriverError('No matching blocks provided for segmentation')

        if self._estimate_tokens(passage=document) <= self.max_segment_tokens:
            LOGGER.debug('Document %s is small enough, no need to segment', document)
            return [document]

        LOGGER.debug('Segmenting document: %s', document)

        tokens = MarkdownIt().parse(document)
        all_levels = sorted(set(int(tok.tag[1]) for tok in tokens if tok.type == 'heading_open'))

        if not all_levels:
            LOGGER.debug('No heading levels found in document, falling back to paragraph split')
            return self._fallback_segment(document, matching_blocks)

        split_level = current_level or all_levels[0]
        blocks = self._split_by_heading_level(tokens, split_level)

        if len(blocks) == 1:
            LOGGER.debug('Only one heading level found, try a deeper level split')
            return self._try_deeper_split(
                document=document,
                matching_blocks=matching_blocks,
                all_levels=all_levels,
                current_level=split_level,
            )

        LOGGER.info('Splitting document by heading level %d', split_level)
        LOGGER.debug('Blocks: %s', blocks)

        return self._match_blocks(blocks, matching_blocks, split_level)

    def _split_by_heading_level(self,
                                tokens: list[Token],
                                split_level: int,
                                ) -> list[tuple[str, str]]:
        """
        Split the document by heading level

        :param tokens: The tokens of the document, parsed by MarkdownIt
        :param split_level: The heading level to split by
        :return: A list of tuples, each containing the title and content
        """
        blocks = []
        current_title = ''
        current_content = []

        def flush_block():
            if current_title or current_content:
                blocks.append((current_title, self._tokens_to_text(current_content)))
                current_content.clear()

        for idx, tok in enumerate(tokens):
            if tok.type == 'heading_open' and int(tok.tag[1]) == split_level:
                flush_block()
                if idx + 1 < len(tokens):
                    current_title = tokens[idx + 1].content
            else:
                current_content.append(tok)

        flush_block()
        return blocks

    @staticmethod
    def _tokens_to_text(tokens: list[Token]) -> str:
        """
        Convert a list of tokens to text

        :param tokens: The tokens to convert
        :return: The text in Markdown format
        """
        return ''.join(getattr(t, 'content', '') for t in tokens).strip()

    def _fallback_segment(self,
                          document: str,
                          matching_blocks: list[str],
                          ) -> list[str]:
        """
        Fallback method for segmenting the document when no headers are found

        :param document: The document to segment, in Markdown format
        :param matching_blocks: The block from Qdrant, which was used to fetch the full
                                content of the note
        :return: The segmented document, in Markdown format
        """
        split = self._fallback_split(document)
        matches = {t for b in matching_blocks for t in split if b in t}
        return list(matches)

    def _try_deeper_split(self,
                          document: str,
                          matching_blocks: list[str],
                          all_levels: list[int],
                          current_level: int,
                          ) -> list[str]:
        """
        Try to split the document by a deeper heading level

        :param document: The document to segment, in Markdown format
        :param matching_blocks: The block from Qdrant, which was used to fetch the full
        :param all_levels: The list of all heading levels found in the document
        :param current_level: The current heading level used for splitting
        :return: The segmented document, in Markdown format
        """
        deeper_levels = [lvl for lvl in all_levels if lvl > current_level]

        if deeper_levels:
            return self._segment_document(
                document=document,
                matching_blocks=matching_blocks,
                current_level=deeper_levels[0],
            )

        return self._fallback_segment(document, matching_blocks)

    def _match_blocks(self,
                      blocks: list[tuple[str, str]],
                      matching_blocks: list[str],
                      current_level: int,
                      ) -> list[str]:
        """
        Match the blocks with the matching blocks from Qdrant
        :param blocks: The blocks to match, in Markdown format
        :param matching_blocks: The block from Qdrant, which was used to fetch the full
                                content of the note
        :param current_level: The current heading level used for splitting
        :return: The segmented document, in Markdown format
        """
        block_texts = []
        matching_set = set(matching_blocks)

        for title, text in blocks:
            combined = f"{title}\n{text}"
            if any(b in combined for b in matching_set):
                if self._estimate_tokens(passage=text) > self.max_segment_tokens:
                    block_texts.extend(
                        self._segment_document(
                            document=text,
                            matching_blocks=[b for b in matching_blocks if b in combined],
                            current_level=current_level,
                        )
                    )
                else:
                    block_texts.append(text)

        LOGGER.debug('Segmented document: %s', block_texts)
        return block_texts

    def _fallback_split(self,
                        document: str,
                        ) -> list[str]:
        """
        Split a document by paragraph if no headers are usable.

        This is a fallback method for splitting the document
        when no headers are found or the document is too long to be accepted.
        :param document: The document to split
        """
        paragraphs = [p.strip() for p in document.split('\n\n') if p.strip()]
        segments = []
        current = ''

        for p in paragraphs:
            tentative = f"{current}\n\n{p}" if current else p
            if self._estimate_tokens(tentative) <= self.max_segment_tokens:
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
                          ) -> list[str]:
        """
        Get the context for the given query

        :param query: The user message
        :param limit: The number of search results to use
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
            ))

        segments = list(set(segments))

        LOGGER.debug('Segments: %s', segments)

        return segments[:limit * 2]

    async def build_prompt(self,
                           query: str,
                           limit: int = 3,
                           ) -> str:
        """
        Construct the prompt using the search results

        This prompt is used to generate the completion
        with the user message

        :param query: The user message
        :param limit: The number of search results to use
        :return: The prompt, added with the search results
        """
        contexts = await self.get_context(
            query=query,
            limit=limit,
        )

        prompt = 'Additional context:\n\n'
        prompt += '\n\n'.join(contexts)
        prompt += 'Question: ' + query
        prompt += '\n\nAnswer: \n\n'

        return prompt
