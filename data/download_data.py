
""" A file that converts .pdf to .md so that it becomes easier for the RAG to learn"""

__author__ = "Majd Jamal"

import csv
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests
from tqdm import tqdm

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


DATA_ROOT = Path('data')          # run from the repo root, like ingest.py
TEMP_DIR = DATA_ROOT / 'temp'

# Maps each manifest category to its target markdown folder.
# 'remuneration' has no folder of its own — filed with agm_egm (it stays
# tagged category='remuneration' in the manifest, so no metadata is lost).
CATEGORY_DIRS = {
    'annual_report':             DATA_ROOT / 'annual_reports',
    'agm_egm':                   DATA_ROOT / 'agm_egm',
    'corporate_governance':      DATA_ROOT / 'governance',
    'sustainability_disclosure': DATA_ROOT / 'sustainability',
    'remuneration':              DATA_ROOT / 'agm_egm',
}

CONVERTED_DIRS = list(set(CATEGORY_DIRS.values()))

MANIFEST_CSV = DATA_ROOT / 'documents.csv'


def load_manifest(csv_path: Path = MANIFEST_CSV) -> list[dict]:
    """ Opens the csv file and reads the downloadable links.

    :param csv_path: Path to the documents manifest CSV
    :return records: List of row dicts, one per document, with a usable PDF link
    """

    # utf-8-sig strips the BOM the exported CSV carries on the first header.
    with open(csv_path, encoding = 'utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    # Keep only rows that actually point at a downloadable PDF.
    records = [r for r in rows
               if r.get('link', '').strip() and r.get('file_format', '').upper() == 'PDF']

    return records


def download_pdf(record: dict,
                 temp_dir: Path = TEMP_DIR,
                 converted_dirs: list[Path] = CONVERTED_DIRS) -> Path | None:
    """ Downloads one PDF into temp_dir, unless it was already converted to markdown.

    The download is skipped when a markdown file with the same name (<stem>.md)
    already exists in any of the converted folders, so re-runs only fetch what
    has not been processed yet.

    :param record: One manifest row; must contain a 'link' to the PDF
    :param temp_dir: Folder where new PDFs are downloaded
    :param converted_dirs: Folders checked for an already-converted <stem>.md
    :return pdf_path: Local path to the PDF in temp, or None if skipped or failed
    """

    link = record['link']

    # Filename comes from the URL itself (unquote handles %20 etc.).
    stem = Path(unquote(urlparse(link).path)).stem

    # Skip if this document was already converted to markdown in any folder.
    for d in converted_dirs:
        if (d / f'{stem}.md').exists():
            return None

    temp_dir.mkdir(parents = True, exist_ok = True)
    pdf_path = temp_dir / f'{stem}.pdf'

    # Skip if the PDF is already waiting in temp from an earlier run.
    if pdf_path.exists():
        return pdf_path

    try:
        response = requests.get(link, timeout = 30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f'  Failed {stem}: {type(e).__name__}: {e}')
        return None

    pdf_path.write_bytes(response.content)

    return pdf_path


def convert_to_markdown(record: dict,
                        pdf_path: Path,
                        converter,
                        category_dirs: dict = CATEGORY_DIRS) -> Path | None:
    """ Converts one downloaded PDF to markdown and files it under its doc-type folder.

    :param record: One manifest row; must contain 'category'
    :param pdf_path: Local path to the downloaded PDF (as returned by download_pdf)
    :param converter: A marker PdfConverter, created once and reused across calls
    :param category_dirs: Mapping of manifest category to its markdown folder
    :return md_path: Path to the written markdown file, or None if skipped or failed
    """

    category = record['category']

    out_dir = category_dirs.get(category)
    if out_dir is None:
        print(f'  Skipped {pdf_path.stem}: unknown category {category!r}')
        return None

    out_dir.mkdir(parents = True, exist_ok = True)
    md_path = out_dir / f'{pdf_path.stem}.md'

    # Idempotent: don't re-convert what is already filed.
    if md_path.exists():
        return md_path

    try:
        rendered = converter(str(pdf_path))
        text, _, _ = text_from_rendered(rendered)
    except Exception as e:
        print(f'  Failed {pdf_path.stem}: {type(e).__name__}: {e}')
        return None

    md_path.write_text(text, encoding = 'utf-8')

    return md_path


def main():
    """ Loads the documents.csv and downloads files accordingly"""

    ### Load file

    records = load_manifest()
    print(f'\n =-=-=-=- Loaded {len(records)} documents from documents.csv -=-=-=-= \n')

    converter = PdfConverter(artifact_dict = create_model_dict())   # loads models once; slow on CPU

    n_done = 0

    for record in tqdm(records, desc = 'Processing', colour = 'cyan', unit = 'pdf'):

        ### download to pdf

        pdf_path = download_pdf(record)

        ### convert to markdown

        if pdf_path:
            if convert_to_markdown(record, pdf_path, converter):
                n_done += 1

    print(f'\nDone. {n_done} markdown files written.')


if __name__ == "__main__":
    main()
