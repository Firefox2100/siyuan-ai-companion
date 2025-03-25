import os
import json
import re

from siyuan_ai_companion.consts import SIYUAN_DATA_DIR
from .note import Note

class Notebook:
    """
    A SiYuan notebook object. It is a folder in the data directory.
    """
    def __init__(self,
                 notebook_id: str,
                 ):
        self.notebook_id = notebook_id

        self.notebook_name = ''

        self._notes: list[Note] = []
        self._loaded = False

    @property
    def notes(self) -> list[Note]:
        """
        The notes in the notebook

        Can only be set by the load method

        :return: The notes in the notebook
        """
        return self._notes

    @property
    def loaded(self) -> bool:
        """
        Whether the notebook has been loaded

        Can only be set by the load method

        :return: True if the notebook has been loaded
        """
        return self._loaded

    def load(self):
        """
        Load the notebook from the data directory
        """
        notebook_path = os.path.join(SIYUAN_DATA_DIR, self.notebook_id)
        metadata_path = os.path.join(notebook_path, '.siyuan', 'conf.json')

        with open(metadata_path, encoding='utf-8') as metadata_file:
            metadata = json.load(metadata_file)
            self.notebook_name = metadata['name']

        # Notes are datetime-random_string.sy files in the notebook directory
        # Iterate the directory to find all notes
        note_pattern = r'^\d{14}-[a-zA-Z0-9]{7}\.sy$'
        for note_file in os.listdir(notebook_path):
            if re.match(note_pattern, note_file):
                note = Note(
                    note_id=note_file,
                    notebook_id=self.notebook_id,
                )
                note.load()

                self._notes.append(note)
