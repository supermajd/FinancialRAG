
"""db.py: SQLite connection, schema setup, and read/write helpers for documents and chunks."""

__author__ = "Majd Jamal"

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    company         TEXT,
    category        TEXT,
    meeting_type    TEXT,
    year            INTEGER,
    document_title  TEXT,
    source_url      TEXT,
    source_page     TEXT,
    md_path         TEXT,
    text            TEXT,
    ingested_at     TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     INTEGER PRIMARY KEY,   -- also the FAISS/BM25 id
    doc_id       TEXT REFERENCES documents(doc_id),
    chunk_index  INTEGER,
    heading      TEXT,
    text         TEXT,
    token_count  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
CREATE INDEX IF NOT EXISTS idx_documents_year     ON documents(year);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


def connect_to_db(db_path: Path) -> sqlite3.Connection:
    """ Opens a SQLite connection with foreign keys enabled.
    :param db_path: Path to the SQLite database file
    :return conn: SQLite connection with row_factory set for dict-like row access
    """

    db_path = Path(db_path)
    db_path.parent.mkdir(parents = True, exist_ok = True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Foreign keys are off by default in SQLite; enable to enforce doc_id references.
    conn.execute('PRAGMA foreign_keys = ON;')

    return conn


def tables_exist(conn: sqlite3.Connection) -> bool:
    """ Checks if the core tables are already present.
    :param conn: SQLite connection
    :return exists: True if the documents table exists
    """

    cursor = conn.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type = 'table' AND name IN ('documents')
        """)

    exists = cursor.fetchone() is not None

    return exists


def create_tables(conn: sqlite3.Connection) -> None:
    """ Creates tables and indexes if they do not already exist.
    :param conn: SQLite connection
    """

    if tables_exist(conn):
        print('Database schema already exists, skipping setup.')
        return

    conn.executescript(SCHEMA)
    conn.commit()


def insert_document(conn: sqlite3.Connection, document) -> None:
    """ Inserts or replaces one parsed document in the documents table.
    INSERT OR REPLACE keyed on the stable doc_id makes re-ingest idempotent.
    :param conn: SQLite connection
    :param document: Document dataclass instance from DocumentParser
    """

    if not document.doc_id:
        raise ValueError('Document is missing doc_id and cannot be stored.')

    ingested_at = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT OR REPLACE INTO documents (
            doc_id, company, category, meeting_type, year,
            document_title, source_url, source_page, md_path, text,
            ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document.doc_id,
            document.company,
            document.category,
            document.meeting_type,
            document.year,
            document.document_title,
            document.source_url,
            document.source_page,
            str(document.md_path),
            document.text,
            ingested_at,
        ),
    )

    conn.commit()


def get_documents(
        conn: sqlite3.Connection,
        doc_id: str = None,
        category: str = None,
        year: int = None,
        limit: int = None) -> list:
    """ Reads documents from the database, optionally filtered.

    :param conn: SQLite connection
    :param doc_id: If given, return the single matching document
    :param category: If given, restrict to one document category
    :param year: If given, restrict to one year
    :param limit: Maximum number of rows to return
    :return documents: List of document dicts, one per row
    """

    if doc_id is not None:
        cursor = conn.execute('SELECT * FROM documents WHERE doc_id = ?', (doc_id,))
        return [dict(row) for row in cursor.fetchall()]

    clauses = []
    params = []

    if category is not None:
        clauses.append('category = ?')
        params.append(category)

    if year is not None:
        clauses.append('year = ?')
        params.append(year)

    where = f' WHERE {" AND ".join(clauses)}' if clauses else ''
    sql = f'SELECT * FROM documents{where} ORDER BY year DESC, category'

    if limit is not None:
        sql += ' LIMIT ?'
        params.append(limit)

    cursor = conn.execute(sql, params)
    documents = [dict(row) for row in cursor.fetchall()]

    return documents
   
   

def reset_documents(conn: sqlite3.Connection, confirm: bool = False) -> int:
    """ Deletes every row from the documents table.
    :param conn: SQLite connection
    :param confirm: Must be True to actually delete. Guard against accidents.
    :return n_deleted: Number of document rows deleted
    """

    if not confirm:
        raise ValueError(
            'reset_documents requires confirm=True. '
            'This will delete all documents.')

    cursor = conn.execute('DELETE FROM documents')

    n_deleted = cursor.rowcount

    conn.commit()
    conn.execute('VACUUM')

    print(f'Reset complete. Deleted {n_deleted} documents.')

    return n_deleted

def insert_chunk(conn: sqlite3.Connection, chunk) -> int:
    """ Inserts one chunk and returns its assigned id.

    chunk_id is left to SQLite (INTEGER PRIMARY KEY); the returned value is reused
    as the FAISS/BM25 id so a search hit maps straight back to this row.
    Does NOT commit — call once after the build loop (thousands of chunks).
    :param conn: SQLite connection
    :param chunk: Chunk instance with doc_id, chunk_index, heading, text, token_count
    :return chunk_id: The integer id assigned to the stored chunk
    """

    cursor = conn.execute(
        """
        INSERT INTO chunks (
            doc_id, chunk_index, heading, text, token_count
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            chunk.doc_id,
            chunk.chunk_index,
            chunk.heading,
            chunk.text,
            chunk.token_count,
        ),
    )

    return cursor.lastrowid

def get_chunks(
        conn: sqlite3.Connection,
        chunk_id: int = None,
        chunk_ids: list = None,
        doc_id: str = None,
        limit: int = None) -> list:
    """ Reads chunks joined to their document metadata.

    Mode is selected by the argument combination:
        - chunk_id given:  the single matching chunk
        - chunk_ids given: the chunks with those ids (retrieval hits)
        - doc_id given:    every chunk of one document, in order
        - no arguments:    all chunks in id order (the index-build read)

    :param conn: SQLite connection
    :param chunk_id: A single chunk id to fetch
    :param chunk_ids: A list of chunk ids to fetch (e.g. retrieval results)
    :param doc_id: Restrict to one document's chunks
    :param limit: Maximum number of rows to return
    :return chunks: List of chunk dicts, each joined with document metadata
    """

    base = """
        SELECT
            c.chunk_id, c.doc_id, c.chunk_index, c.heading, c.text, c.token_count,
            d.category, d.year, d.document_title, d.source_url
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
    """

    if chunk_id is not None:
        cursor = conn.execute(base + ' WHERE c.chunk_id = ?', (chunk_id,))
        return [dict(row) for row in cursor.fetchall()]

    if chunk_ids is not None:
        if not chunk_ids:
            return []
        placeholders = ','.join('?' for _ in chunk_ids)
        cursor = conn.execute(base + f' WHERE c.chunk_id IN ({placeholders})', tuple(chunk_ids))
        return [dict(row) for row in cursor.fetchall()]

    if doc_id is not None:
        cursor = conn.execute(base + ' WHERE c.doc_id = ? ORDER BY c.chunk_index', (doc_id,))
        return [dict(row) for row in cursor.fetchall()]

    sql = base + ' ORDER BY c.chunk_id'
    params = ()

    if limit is not None:
        sql += ' LIMIT ?'
        params = (limit,)

    cursor = conn.execute(sql, params)
    chunks = [dict(row) for row in cursor.fetchall()]

    return chunks


def reset_chunks(conn: sqlite3.Connection, confirm: bool = False) -> int:
    """ Deletes every row from the chunks table.

    Run before re-indexing so chunk ids restart and stay aligned with a freshly
    rebuilt FAISS/BM25 index.
    :param conn: SQLite connection
    :param confirm: Must be True to actually delete. Guard against accidents.
    :return n_deleted: Number of chunk rows deleted
    """

    if not confirm:
        raise ValueError(
            'reset_chunks requires confirm=True. '
            'This will delete all chunks.')

    cursor = conn.execute('DELETE FROM chunks')

    n_deleted = cursor.rowcount

    conn.commit()
    conn.execute('VACUUM')

    print(f'Reset complete. Deleted {n_deleted} chunks.')

    return n_deleted
