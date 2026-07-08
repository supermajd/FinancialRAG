
"""retrieve.py: Hybrid retrieval over BM25 and vector search for document chunks."""

__author__ = "Majd Jamal"

import numpy as np

from app.index import FAISS, BM25, SentenceBert

class Retrieval:

    def __init__(self):

        #-=-=-=-
        self.faiss = FAISS()
        self.faiss.load()

        #-=-=-=-
        self.bm25 = BM25()
        self.bm25.load()

        self.embedder = SentenceBert()

        self.rrf_k = 60


    def transform(self, text: str) -> np.ndarray:
        """ Embeds a single query string for vector search.
        :param text: User query string
        :return embedding: Array with shape (1, Ndim), float32
        """

        embedding = self.embedder.transform([text])
        return embedding
    
    def fuse(self, bm25_res, faiss_res):
        """ Combines BM25 and FAISS result lists via reciprocal rank fusion.
        Each chunk's fused score is the sum of 1 / (k + rank) across both lists,
        so chunks ranked highly by either retriever bubble to the top.
        :param bm25_res: BM25 results, list of (chunk, meta, score)
        :param faiss_res: FAISS results, list of (chunk, meta, score)
        :return fused: List of (chunk, meta, fused_score), sorted by score desc
        """

        scores = {}
        records = {}
        sims = {}

        # FAISS pass: keep the cosine similarity (inner product of L2-normalized
        # vectors) so the API can show an interpretable 0-1 match strength.
        for rank, (chunk, meta, sim) in enumerate(faiss_res):
            chunk_id = meta['chunk_id']
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            records[chunk_id] = (chunk, meta)
            sims[chunk_id] = float(sim)
        # BM25 pass:
        for rank, (chunk, meta, _) in enumerate(bm25_res):
            chunk_id = meta['chunk_id']
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            records[chunk_id] = (chunk, meta)

        fused = []
        for cid, score in sorted(scores.items(), key = lambda x: x[1], reverse = True):
            chunk, meta = records[cid]
            # Cosine similarity (0.0 for BM25-only hits that had no vector match).
            meta = {**meta, 'similarity': sims.get(cid, 0.0)}
            fused.append((chunk, meta, score))

        return fused

    def _match(self, meta: dict, filters: dict) -> bool:
        """ Checks whether a chunk's metadata satisfies all active filters.
        :param meta: Chunk metadata dict
        :param filters: Filter dict with optional 'year' and 'category'
        :return matched: True if every filter passes
        """

        if 'year' in filters and str(meta.get('year')) != str(filters['year']):
            return False

        if 'category' in filters and meta.get('category') != filters['category']:
            return False

        return True
        
    def search(self,
            text: str,
            k: int = 10,
            k_each: int = 30,
            filters: dict = None) -> list:
        """ Hybrid search with optional metadata filters.
        :param text: User query string
        :param k: Number of final results to return
        :param k_each: Number of results to fetch from each retriever before fusion
        :param filters: Optional dict of metadata filters. Supported keys:
                        'year'      -> restrict to one report year (e.g. 2019)
                        'category'  -> restrict to one document category
                                       (e.g. 'annual_report', 'agm_egm')
        :return results: List of (chunk, metadata, score) tuples
        """

        embedding = self.transform(text)

        bm25_match = self.bm25.search(text, k = k_each)
        faiss_match = self.faiss.search(embedding, k = k_each)

        fused = self.fuse(bm25_match, faiss_match)
        
        if filters:

            filtered = []

            for item in fused:
                chunk, meta, score = item

                if self._match(meta, filters):
                    filtered.append(item)

            fused = filtered

        return fused[:k]


if __name__ == "__main__":

    retriever = Retrieval()

    query = 'dividend per share proposed by the board'

    filters = {
        'year':     2019,
        'category': 'agm_egm',
    }

    results = retriever.search(query, k = 7, k_each = 20, filters = filters)

    print(f'\n =-=-=-=- Top {len(results)} results for: {query!r} -=-=-=-= \n')

    for chunk, meta, score in results:
        print(
            f'[{score:.4f}] '
            f'chunk {meta["chunk_id"]} | '
            f'{meta["year"]} | '
            f'{meta["document_title"]} | '
            f'{chunk[:80]}...')