from enum import Enum

from siyuan_ai_companion.errors import UnknownNodeTypeError


class NoteContentType(Enum):
    PARAGRAPH = 'NodeParagraph'
    BLOCKQUOTE = 'NodeBlockquote'
    BLOCKQUOTE_MARKER = 'NodeBlockquoteMarker'
    TEXT = 'NodeText'
    TEXT_MARK = 'NodeTextMark'
    SUPER_BLOCK = 'NodeSuperBlock'
    SUPER_BLOCK_OPEN_MARKER = 'NodeSuperBlockOpenMarker'
    SUPER_BLOCK_CLOSE_MARKER = 'NodeSuperBlockCloseMarker'
    SUPER_BLOCK_LAYOUT_MARKER = 'NodeSuperBlockLayoutMarker'
    HEADING = 'NodeHeading'
    LIST = 'NodeList'
    LIST_ITEM = 'NodeListItem'
    THEMATIC_BREAK = 'NodeThematicBreak'


class NoteContent:
    def __init__(self,
                 content: str,
                 ):
        self.content = content

    @classmethod
    def _parse(cls, payload: dict) -> str:
        """
        Parse the JSON note content (a block) into plain text

        :param payload: The JSON note content
        :return: The plain text content
        """
        try:
            content_type = NoteContentType(payload['Type'])
        except ValueError as e:
            raise UnknownNodeTypeError(f'Unknown node type {payload["Type"]}') from e

        if content_type in [
            NoteContentType.BLOCKQUOTE_MARKER,
            NoteContentType.SUPER_BLOCK_OPEN_MARKER,
            NoteContentType.SUPER_BLOCK_CLOSE_MARKER,
            NoteContentType.SUPER_BLOCK_LAYOUT_MARKER,
            NoteContentType.THEMATIC_BREAK,
        ]:
            return ''

        if content_type == NoteContentType.PARAGRAPH:
            children = [cls._parse(c) for c in payload['Children']]

            return f'{"".join(children)}\n'

        if content_type == NoteContentType.BLOCKQUOTE:
            children = [cls._parse(c) for c in payload['Children']]

            return f'\n{"".join(children)}\n'

        if content_type == NoteContentType.TEXT:
            return payload['Data']

        if content_type == NoteContentType.TEXT_MARK:
            return payload['TextMarkTextContent']

        if content_type == NoteContentType.SUPER_BLOCK:
            children = [cls._parse(c) for c in payload['Children']]

            return f'\n{"".join(children)}\n'

        if content_type == NoteContentType.HEADING:
            heading_level = payload['HeadingLevel']
            children = [cls._parse(c) for c in payload['Children']]

            return f'\n{"#" * heading_level} {"".join(children)}\n'

        if content_type == NoteContentType.LIST:
            children = [cls._parse(c) for c in payload['Children']]

            return f'{f"{chr(10)} - ".join(children)}\n'

        if content_type == NoteContentType.LIST_ITEM:
            children = [cls._parse(c) for c in payload['Children']]

            return f'{"".join(children)}'

        raise UnknownNodeTypeError(f'Cannot parse {content_type}')

    @classmethod
    def parse(cls, payload: dict):
        """
        Construct a NoteContent object from a JSON note content

        :param payload: The JSON note content
        """
        content = cls._parse(payload)

        return cls(content)
