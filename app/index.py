
"""index.py: Builds the FAISS and BM25 indexes from parsed document chunks."""

__author__ = "Majd Jamal"


import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
import pickle
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.db import connect_to_db

TARGET_TOKENS = 256     # chunk content target; leaves headroom under the 384 ceiling
OVERLAP_TOKENS = 40     # token overlap between consecutive prose chunks
MIN_TOKENS = 16         # drop fragments smaller than this


TOKEN_RE = re.compile(r'\w+', flags = re.UNICODE)

INVISIBLE_CHARS = (
    '\u200b'  # zero-width space
    '\u200c'  # zero-width non-joiner
    '\u200d'  # zero-width joiner
    '\u00ad'  # soft hyphen
    '\ufeff'  # BOM
)

INVISIBLE_RE = re.compile(f'[{INVISIBLE_CHARS}]')
WHITESPACE_RE = re.compile(r'\s+')


@dataclass
class Chunk:

    doc_id: str
    chunk_index: int
    heading: str
    text: str
    token_count: int

class SmartChunker:
    """
    Markdown-aware chunker for financial reports. Splits along structure
    (headings, tables, paragraphs) rather than blind windows, sizes by the
    embedder's tokenizer, and keeps tables intact.
    """

    def __init__(self,
                model_name: str = 'KBLab/sentence-bert-swedish-cased',
                target_tokens: int = TARGET_TOKENS,
                overlap_tokens: int = OVERLAP_TOKENS,
                min_tokens: int = MIN_TOKENS) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

        
    def transform(self, text: str, doc_id: str) -> list:
        """ Entry point: segment the markdown, then pack it into Chunks.
        :param text: The document's markdown text
        :param doc_id: doc_id of the parent document
        :return chunks: List of Chunk instances
        """

        if not text or not text.strip():
            return []

        segments = self._segment(text)

        return self._pack(segments, doc_id)

    def _segment(self, text: str) -> list:
        """ Walks the markdown line by line, grouping it into ordered blocks.

        Each block is (heading, kind, content): a run of table rows becomes a
        'table' block, runs of prose become 'prose' blocks, and every block is
        tagged with the nearest heading above it.
        :param text: The document's markdown text
        :return segments: List of (heading, kind, content) tuples in reading order
        """

        segments = []
        heading = ''
        prose, table = [], []

        def flush_prose():
            block = '\n'.join(prose).strip()
            if block:
                segments.append((heading, 'prose', block))
            prose.clear()

        def flush_table():
            block = '\n'.join(table).strip()
            # A pipe run is only a real table if it contains a delimiter row;
            # otherwise it was prose that happened to start with '|'.
            if block and any(self._is_delimiter(line) for line in table):
                segments.append((heading, 'table', block))
            elif block:
                segments.append((heading, 'prose', block))
            table.clear()

        for raw_line in text.splitlines():

            line = raw_line.rstrip()
            stripped = line.strip()

            # Heading: close any open blocks, then switch the current heading.
            if stripped.startswith('#'):
                flush_table()
                flush_prose()
                heading = stripped.lstrip('#').strip()
                continue

            # Pipe line: part of a table run.
            if stripped.startswith('|'):
                flush_prose()
                table.append(line)
                continue

            # Any non-pipe line ends an open table run.
            if table:
                flush_table()

            # Blank line ends a prose paragraph.
            if not stripped:
                flush_prose()
                continue

            prose.append(line)

        flush_table()
        flush_prose()

        return segments

    def _pack(self, segments: list, doc_id: str) -> list:
        """ Packs segments into Chunks within budget and within one heading.

        Prose blocks are merged into a running buffer until they near the token
        target; tables get their own chunk(s); over-budget blocks are split. A
        chunk never spans two headings.
        :param segments: Output of _segment — (heading, kind, content) tuples
        :param doc_id: doc_id of the parent document
        :return chunks: List of Chunk instances in reading order
        """

        chunks = []
        current_heading = None
        current_chunk = ''

        def emit(text, keep_small = False):
            # Seal one piece of text into a Chunk, unless it's empty/too small.
            text = text.strip()
            if not text:
                return
            if not keep_small and self._count(text) < self.min_tokens:
                return
            chunks.append(self._make_chunk(doc_id, len(chunks), current_heading or '', text))

        def flush():
            # Seal whatever prose has accumulated in the buffer.
            nonlocal current_chunk
            emit(current_chunk)
            current_chunk = ''

        for heading, kind, content in segments:

            # Section boundary: never let a chunk straddle two headings.
            if heading != current_heading:
                flush()
                current_heading = heading

            if kind == 'table':
                flush()                                      # tables never merge with prose
                for piece in self._split_table(content):
                    emit(piece, keep_small = True)           # keep even small tables
                continue

            # Prose.
            if self._count(content) > self.target_tokens:    # rare: one giant paragraph
                flush()
                for piece in self._split_prose(content):
                    emit(piece)
                continue

            candidate = f'{current_chunk}\n\n{content}'.strip() if current_chunk else content

            if self._count(candidate) <= self.target_tokens:
                current_chunk = candidate                     # fits → merge
            else:
                tail = self._overlap_tail(current_chunk)      # capture overlap BEFORE flushing
                flush()
                current_chunk = f'{tail}\n\n{content}'.strip() if tail else content

        flush()                                               # don't lose the last buffer

        return chunks


    def _split_table(self, table: str) -> list:
        """ Splits an over-budget table into row-groups, repeating the header.
        :param table: Markdown table text
        :return pieces: List of table strings, each within the token budget
        """

        if self._count(table) <= self.target_tokens:
            return [table]

        lines = table.splitlines()
        header = lines[:2]          # GFM table: column-label row + the |---| row
        body = lines[2:]

        pieces = []
        group = list(header)

        for row in body:

            # Seal the group before it overflows, then restart with the header.
            if len(group) > len(header) and self._count('\n'.join(group + [row])) > self.target_tokens:
                pieces.append('\n'.join(group))
                group = list(header)

            group.append(row)

        pieces.append('\n'.join(group))

        return pieces

    def _split_prose(self, text: str) -> list:
        """ Splits an over-budget paragraph into overlapping token windows.
        :param text: A single oversize prose block
        :return pieces: List of text windows, each within budget
        """

        ids = self.tokenizer.encode(text, add_special_tokens = False)

        if len(ids) <= self.target_tokens:
            return [text]

        step = self.target_tokens - self.overlap_tokens
        pieces = []

        for start in range(0, len(ids), step):
            window_ids = ids[start:start + self.target_tokens]
            pieces.append(self.tokenizer.decode(window_ids).strip())
            if start + self.target_tokens >= len(ids):
                break

        return pieces


    def _overlap_tail(self, text: str) -> str:
        """ Returns the last OVERLAP_TOKENS tokens of text, for prose continuity.
        :param text: The chunk text being sealed
        :return tail: Trailing slice to seed the next chunk
        """

        ids = self.tokenizer.encode(text, add_special_tokens = False)

        if len(ids) <= self.overlap_tokens:
            return text

        return self.tokenizer.decode(ids[-self.overlap_tokens:]).strip()

    def _count(self, text: str) -> int:
        """ Token count as the embedder sees it (no special tokens).
        :param text: Any text
        :return n: Number of tokens
        """

        return len(self.tokenizer.encode(text, add_special_tokens = False))


    def _make_chunk(self, doc_id: str, chunk_index: int, heading: str, text: str):
        """ Builds one Chunk, computing its token_count.
        :param doc_id: doc_id of the parent document
        :param chunk_index: Position of this chunk within the document
        :param heading: Section heading the chunk sits under
        :param text: The chunk text
        :return chunk: Populated Chunk instance
        """

        return Chunk(
            doc_id = doc_id,
            chunk_index = chunk_index,
            heading = heading,
            text = text,
            token_count = self._count(text))

    def _is_delimiter(self, line: str) -> bool:
        """ True if a stripped line is a markdown table delimiter row (e.g. |---|:--:|).
        :param line: One markdown line
        :return is_delim: True if the line is a table delimiter row
        """

        stripped = line.strip()

        return bool(stripped) and '-' in stripped and set(stripped) <= set('|-: ')


class SentenceBert:
    """
    Sentence Embedder

    Wraps KB-Lab's Swedish bilingual sentence transformer. Produces 768-dim
    vectors locally without any external API call. Suitable for Swedish and
    English document content.
    """

    def __init__(self,
                 model_name: str = 'KBLab/sentence-bert-swedish-cased',
                 dim: int = 768) -> None:

        self.model_name = model_name
        self.dim = dim
        self.model = SentenceTransformer(model_name)


    def transform(self, texts: list) -> np.ndarray:
        """ Embeds a list of strings.
        :param texts: List of text strings, length Npts
        :return embeddings: Array with shape (Npts, Ndim), float32
        """

        if not texts:
            return np.zeros((0, self.dim), dtype = 'float32')

        embedding = self.model.encode(
            texts,
            batch_size = 32,
            show_progress_bar = True,
            convert_to_numpy = True)

        return embedding.astype('float32')

class FAISS:

    def __init__(self, dim: int = 768, path_to_index: str = 'data/index/') -> None:
        self.dim = dim
        self.index_path = Path(path_to_index)
        self.index_path.mkdir(parents = True, exist_ok = True)

        self.index = None
        self.chunks = None
        self.metadata = None
        
    def fit(self, embeddings: np.ndarray, chunks: list, metadata: list):
        
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(self.dim)
        index.add(embeddings)

        self.index = index
        self.chunks = chunks
        self.metadata = metadata

        print(f'Built FAISS index with {index.ntotal} vectors of dim {self.dim}')

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list:
        """ Returns the top-k chunks for a query embedding.
        :param query_embedding: Array with shape (1, Ndim) or (Ndim,), float32
        :param k: Number of results to return
        :return results: List of (chunk, metadata, score) tuples
        """

        if self.index is None:
            raise ValueError('Index is empty. Call fit() or load() first.')

        # Normalize shape and dtype to match the indexed vectors.
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        results = [
            (self.chunks[i], self.metadata[i], float(scores[0][j]))
            for j, i in enumerate(indices[0])
            if i != -1
        ]

        return results

    def save(self) -> None:
        """ Persists the FAISS index and parallel chunks/metadata to disk.
        """

        if self.index is None:
            raise ValueError('Nothing to save. Call fit() first.')

        faiss.write_index(self.index, str(self.index_path / 'faiss.index'))

        with open(self.index_path / 'chunks.json', 'w', encoding = 'utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii = False)

        with open(self.index_path / 'metadata.json', 'w', encoding = 'utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii = False, default = str)

        print(f'Saved FAISS index to {self.index_path}')

    def load(self) -> None:
        """ Loads a previously saved index, chunks, and metadata into memory.
        """

        faiss_file = self.index_path / 'faiss.index'

        if not faiss_file.exists():
            raise ValueError(f'No FAISS index found at {faiss_file}.')

        self.index = faiss.read_index(str(faiss_file))

        with open(self.index_path / 'chunks.json', 'r', encoding = 'utf-8') as f:
            self.chunks = json.load(f)

        with open(self.index_path / 'metadata.json', 'r', encoding = 'utf-8') as f:
            self.metadata = json.load(f)

        print(f'Loaded FAISS index with {self.index.ntotal} vectors from {self.index_path}')

class BM25:

    def __init__(self, path_to_index: str = 'data/index/') -> None:

        self.index_path = Path(path_to_index)
        self.index_path.mkdir(parents = True, exist_ok = True)

        self.bm25 = None
        self.chunks = None
        self.metadata = None

    def _tokenizer(self, text):
        """ Splits text into lowercase whitespace tokens.
        :param text: Raw chunk text
        :return tokens: List of lowercase tokens
        """

        return TOKEN_RE.findall(text.lower())

    def fit(self, chunks, metadata):
        """ Builds the BM25 index from chunk texts.
        :param chunks: List of chunk texts, length Npts
        :param metadata: List of chunk metadata dicts, length Npts
        """

        tokenized = [self._tokenizer(text) for text in tqdm(chunks, desc = 'Tokenizing for BM25')]

        self.bm25 = BM25Okapi(tokenized)
        self.chunks = chunks
        self.metadata = metadata

        print(f'Built BM25 index over {len(chunks)} chunks')

    def search(self, query: str, k: int = 5) -> list:
        """ Returns the top-k chunks for a keyword query.
        :param query: User query string
        :param k: Number of results to return
        :return results: List of (chunk, metadata, score) tuples
        """

        query_tokens = self._tokenizer(query)
        scores = self.bm25.get_scores(query_tokens)

        top_idx = np.argsort(scores)[-k:][::-1]
        results = [(self.chunks[i], self.metadata[i], float(scores[i])) for i in top_idx]

        return results
    
    def save(self) -> None:
        """ Persists the BM25 index to disk.
        """

        if self.bm25 is None:
            raise ValueError('Nothing to save. Call fit() first.')

        with open(self.index_path / 'bm25.pkl', 'wb') as f:
            pickle.dump(self.bm25, f)

        print(f'Saved BM25 index to {self.index_path}')

    def load(self) -> None:
        """ Loads a previously saved BM25 index into memory.
        """

        bm25_file = self.index_path / 'bm25.pkl'

        if not bm25_file.exists():
            raise ValueError(f'No BM25 index found at {bm25_file}.')

        with open(bm25_file, 'rb') as f:
            self.bm25 = pickle.load(f)

        with open(self.index_path / 'chunks.json', 'r', encoding = 'utf-8') as f:
            self.chunks = json.load(f)

        with open(self.index_path / 'metadata.json', 'r', encoding = 'utf-8') as f:
            self.metadata = json.load(f)

        print(f'Loaded BM25 index over {len(self.chunks)} chunks from {self.index_path}')

def build_indexable_text(document: dict, chunk) -> str:
    """ Prepends document title, type, year, and section heading to a chunk.
    Lets BM25 and the embedder match on the report and section, not just the
    chunk body — a lone table row rarely names its own document, year, or section.
    :param document: Document row with document_title, category, year
    :param chunk: Chunk instance with heading and text
    :return text: Header + chunk text, ready for BM25 and embedding
    """

    title = document['document_title'] or ''
    category = document['category'] or ''
    year = document['year'] or ''
    heading = chunk.heading or ''

    header = (
        f'Title: {title}\n'
        f'Type: {category}  Year: {year}\n'
        f'Section: {heading}\n\n'
    )

    return header + chunk.text



if __name__ == "__main__":
    pass