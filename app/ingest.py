"""ingest.py: Ingests converted markdown documents into the documents table."""

__author__ = "Majd Jamal"

import json
import unicodedata
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
import csv

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, unquote

from bs4 import BeautifulSoup
from tqdm import tqdm

from app.db import connect_to_db, insert_document

DATA_ROOT = Path('data')
MANIFEST_CSV = DATA_ROOT / 'documents.csv'
DB_PATH = DATA_ROOT / 'documents.db'


CATEGORY_DIRS = {
    'annual_report':             DATA_ROOT / 'annual_reports',
    'agm_egm':                   DATA_ROOT / 'agm_egm',
    'corporate_governance':      DATA_ROOT / 'governance',
    'sustainability_disclosure': DATA_ROOT / 'sustainability',
    'remuneration':              DATA_ROOT / 'agm_egm',
}

class DocumentParser:

    def parse(self, md_text: str, record: dict, md_path: str) -> "Document":
        """ Builds one financial Document from its markdown text and manifest row.

        A markdown file carries no headers, so every catalog field comes from the
        manifest record, not the text.
        :param md_text: Converted markdown content of the document
        :param record: One manifest row (category, year, document_title, link, ...)
        :param md_path: Path to the .md file on disk
        :return document_obj: Populated Document instance ready for the database
        """

        link = record['link']

        # Same stem used by download/convert, so doc_id is stable across re-runs.
        stem = Path(unquote(urlparse(link).path)).stem

        document_obj = Document(
            doc_id         = stem,
            company        = record.get('company', ''),
            category       = record['category'],
            meeting_type   = record.get('meeting_type', ''),
            year           = int(record['year']) if record.get('year') else None,
            document_title = self._clean_title(record.get('document_title', ''), record.get('year')),
            source_url     = link,
            source_page    = record.get('source_page', ''),
            md_path        = md_path,
            text           = md_text)

        return document_obj

    def _clean_title(self, title: str, year) -> str:
        """ Normalizes the manifest title: drops the ', PDF' suffix and appends the
        year when the title doesn't already name one.
        :param title: Raw document_title from the manifest
        :param year: Document year from the manifest
        :return title: Cleaned, year-stamped title
        """

        title = title.strip()

        if title.lower().endswith(', pdf'):
            title = title[:-5].strip()

        if year and str(year) not in title:
            title = f'{title} {year}'

        return title


class DocumentReader:
    """ Reads converted markdown documents, pairing each with its manifest row.

    Iteration is driven by the manifest (documents.csv): each row points to one
    converted .md under data/<category>/<stem>.md.
    """

    def __init__(self, manifest_csv: Path, category_dirs: dict) -> None:
        """ :param manifest_csv: Path to documents.csv
            :param category_dirs: Mapping of manifest category to its markdown folder
        """

        self.manifest_csv = Path(manifest_csv)
        self.category_dirs = category_dirs

        if not self.manifest_csv.exists():
            raise ValueError(f'Manifest not found: {self.manifest_csv}')

    def iter_documents(self):
        """ Yields (md_text, record, md_path) for every converted document.
        Rows whose markdown hasn't been produced yet are skipped.
        """

        with open(self.manifest_csv, encoding = 'utf-8-sig') as f:
            records = list(csv.DictReader(f))

        for record in records:

            out_dir = self.category_dirs.get(record['category'])
            if out_dir is None:
                continue

            stem = Path(unquote(urlparse(record['link']).path)).stem
            md_path = out_dir / f'{stem}.md'

            if not md_path.exists():
                continue

            md_text = md_path.read_text(encoding = 'utf-8')

            yield md_text, record, str(md_path)
@dataclass
class Document:

    doc_id: str
    company: str
    category: str
    meeting_type: str
    year: int
    document_title: str
    source_url: str
    source_page: str
    md_path: str
    text: str
    
if __name__ == '__main__':

    #-=-=-=-
    # Setup
    #-=-=-=-

    conn = connect_to_db(DB_PATH)

    reader = DocumentReader(MANIFEST_CSV, CATEGORY_DIRS)
    parser = DocumentParser()

    #-=-=-=-
    # Ingest
    #-=-=-=-

    documents = list(reader.iter_documents())

    print(f'\n =-=-=-=- Ingesting {len(documents)} documents from {MANIFEST_CSV} -=-=-=-= \n')

    n_skipped = 0

    for md_text, record, md_path in tqdm(
            documents,
            desc = 'Ingesting documents',
            colour = 'cyan',
            unit = 'doc'):

        try:
            document_object = parser.parse(md_text, record, md_path)
            insert_document(conn, document_object)
        except Exception as e:
            n_skipped += 1
            print(f'  Skipped {Path(md_path).name}: {type(e).__name__}: {e}')
            continue

    print(f'\nDone. Skipped {n_skipped} documents.')