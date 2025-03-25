import os
import json
import re

from siyuan_ai_companion.consts import SIYUAN_DATA_DIR
from siyuan_ai_companion.errors import FileNotNoteError
from .note_content import NoteContent


class Note:
    """
    A class to represent a note in SiYuan.

    A note is a .sy file with JSON structure, containing
    the note content and metadata.
    """
    def __init__(self,
                 note_id: str,
                 notebook_id: str,
                 parents: list[str] = None,
                 ):
        self.note_id = note_id
        self.notebook_id = notebook_id
        self.parents = parents or []
        self.note_name = ''

        self._content: list[NoteContent] = []
        self._children: list[Note] = []
        self._loaded = False

    @property
    def content(self) -> list[NoteContent]:
        """
        The content of the note

        Can only be set by the load method

        :return: The content of the note
        """
        return self._content

    @property
    def children(self) -> list['Note']:
        """
        The children of the note

        Can only be set by the load method

        :return: The children of the note
        """
        return self._children

    @property
    def loaded(self) -> bool:
        """
        Whether the note has been loaded

        Can only be set by the load method

        :return: True if the note has been loaded
        """
        return self._loaded

    @property
    def plain_text(self) -> str:
        """
        Get the plain text content of the note

        :return: The plain text content of the note
        """
        if not self.loaded:
            self.load()

        content_str = ''.join([c.content for c in self.content])

        # Clean up consecutive newlines
        content_str = re.sub(r'\n{2,}', '\n', content_str)

        return content_str

    def load(self):
        """
        Load the note from the data directory

        :return:
        """
        note_path = os.path.join(
            SIYUAN_DATA_DIR,
            self.notebook_id,
            *self.parents,
            f'{self.note_id}.sy'
        )

        with open(note_path, encoding='utf-8') as note_file:
            note_data = json.load(note_file)

            if note_data['Type'] != 'NodeDocument':
                raise FileNotNoteError(f'File {note_path} is not a note')

            self.note_name = note_data['Properties']['title']

            for block in note_data['Children']:
                content = NoteContent.parse(block)

                self._content.append(content)

        self._loaded = True
