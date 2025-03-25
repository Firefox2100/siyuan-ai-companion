from siyuan_ai_companion.model import Notebook, Note, RagDriver


def index_note(note: Note):
    """
    Index a note and add it to the vector index
    """
    rag_driver = RagDriver()

    rag_driver.add_note(
        note_id=note.note_id,
        note_content=note.plain_text,
    )

    if note.children:
        for child in note.children:
            index_note(child)


def rebuild_all_index():
    """
    Remove and rebuild vector index for all notes
    """
    notebooks = Notebook.list_notebooks()
    rag_driver = RagDriver()

    rag_driver.delete_all()

    for notebook in notebooks:
        notebook.load()

        for note in notebook.notes:
            index_note(note)
