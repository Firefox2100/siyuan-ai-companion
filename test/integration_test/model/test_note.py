import pytest

from siyuan_ai_companion.model.note import Note


class TestNote:
    def test_init(self):
        note = Note(
            note_id='20200812220555-lj3enxa',
            notebook_id='20210808180117-czj9bvb',
        )

        assert note.note_id == '20200812220555-lj3enxa'
        assert note.notebook_id == '20210808180117-czj9bvb'
        assert note.parents == []
        assert not note.content
        assert not note.children
        assert note.loaded is False

    def test_load(self):
        note = Note(
            note_id='20200812220555-lj3enxa',
            notebook_id='20210808180117-czj9bvb',
        )

        note.load()

        plain_text = note.plain_text

        assert bool(plain_text)
