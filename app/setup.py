"""setup.py: One-shot bootstrap. Creates the database and builds all indexes.

Run once after cloning the repo and converting the PDFs with data/download_data.py.
After this finishes, the API can be started with `uvicorn app.api:app`.
"""

__author__ = "Majd Jamal"

from pathlib import Path

import numpy as np
from tqdm import tqdm
import sqlite3

from app.db import (
    connect_to_db,
    create_tables,
    get_documents,
    insert_document,
    insert_chunk,
    reset_chunks,
)
from app.index import BM25, FAISS, SentenceBert, SmartChunker, build_indexable_text
from app.ingest import DocumentReader, DocumentParser, CATEGORY_DIRS, MANIFEST_CSV

DB_PATH = 'data/documents.db'
INDEX_DIR = 'data/index/'

def setup_database(db_path: str):
    """ Opens the SQLite database and creates the schema if missing.
    :param db_path: Path to the SQLite database file
    :return conn: SQLite connection ready for inserts
    """

    conn = connect_to_db(db_path)
    create_tables(conn)

    return conn

def ingest_documents(conn, manifest_csv, category_dirs) -> int:
    """ Reads every converted markdown document and inserts it into the documents table.
    :param conn: SQLite connection
    :param manifest_csv: Path to documents.csv
    :param category_dirs: Mapping of manifest category to its markdown folder
    :return n_ingested: Number of documents successfully inserted
    """

    reader = DocumentReader(manifest_csv, category_dirs)
    parser = DocumentParser()

    documents = list(reader.iter_documents())

    if not documents:
        raise ValueError(
            f'No converted markdown found via {manifest_csv}. '
            f'Run data/download_data.py first to download and convert the PDFs.')

    n_ingested = 0
    n_skipped = 0

    for md_text, record, md_path in tqdm(
            documents,
            desc = 'Ingesting documents',
            colour = 'cyan',
            unit = 'doc'):

        try:
            document_object = parser.parse(md_text, record, md_path)
            insert_document(conn, document_object)
            n_ingested += 1
        except Exception as e:
            n_skipped += 1
            print(f'  Skipped {Path(md_path).name}: {type(e).__name__}: {e}')
            continue

    print(f'Ingested {n_ingested} documents. Skipped {n_skipped}.')

    return n_ingested

    
def build_indexes(conn, index_dir: str) -> None:
    """ Chunks every document, embeds, and writes FAISS + BM25 indexes to disk.
    :param conn: SQLite connection
    :param index_dir: Directory where index files will be written
    """

    reset_chunks(conn, confirm = True)

    chunker = SmartChunker()
    embedder = SentenceBert()
    faiss_index = FAISS(dim = embedder.dim, path_to_index = index_dir)

    documents = get_documents(conn)

    chunk_texts = []
    all_metadata = []

    for document in tqdm(documents, desc = 'Chunking documents'):

        chunks = chunker.transform(document['text'], document['doc_id'])

        for chunk_object in chunks:

            chunk_id = insert_chunk(conn, chunk_object)            # persist + get its id
            indexable_text = build_indexable_text(document, chunk_object)

            chunk_texts.append(indexable_text)
            all_metadata.append({
                'chunk_id':       chunk_id,
                'doc_id':         document['doc_id'],
                'chunk_index':    chunk_object.chunk_index,
                'heading':        chunk_object.heading,
                'category':       document['category'],
                'meeting_type':   document['meeting_type'],
                'year':           document['year'],
                'document_title': document['document_title'],
                'source_url':     document['source_url'],
            })

    conn.commit()                              # insert_chunk doesn't commit per row

    print(f'Created {len(chunk_texts)} chunks from {len(documents)} documents')

    embeddings = embedder.transform(chunk_texts)

    Path(index_dir).mkdir(parents = True, exist_ok = True)
    np.save(Path(index_dir) / 'embeddings.npy', embeddings)

    faiss_index.fit(embeddings, chunk_texts, all_metadata)
    faiss_index.save()

    bm25_index = BM25(path_to_index = index_dir)
    bm25_index.fit(chunk_texts, all_metadata)
    bm25_index.save()

def main() -> None:

    #-=-=-=-
    # Setup database
    #-=-=-=-

    print('\n =-=-=-=- Setting up database -=-=-=-= \n')

    conn = setup_database(DB_PATH)

    #-=-=-=-
    # Ingest documents
    #-=-=-=-

    print(f'\n =-=-=-=- Ingesting documents via {MANIFEST_CSV} -=-=-=-= \n')

    n_ingested = ingest_documents(conn, MANIFEST_CSV, CATEGORY_DIRS)

    if n_ingested == 0:
         raise ValueError('No documents were ingested. Nothing to index.')

    #-=-=-=-
    # Build indexes
    #-=-=-=-

    print(f'\n =-=-=-=- Building indexes in {INDEX_DIR} -=-=-=-= \n')

    build_indexes(conn, INDEX_DIR)

    #-=-=-=-
    # Done
    #-=-=-=-

    print('\n =-=-=-=- Setup complete -=-=-=-= \n')
    print(f'Database: {DB_PATH}')
    print(f'Indexes:  {INDEX_DIR}')
    print('You can now start the API with: uvicorn app.api:app')

if __name__ == '__main__':
    main()