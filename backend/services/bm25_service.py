import json
import pickle
from pathlib import Path
from typing import List

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:
        """Small dependency-free fallback with the same methods used here."""
        def __init__(self, corpus):
            self.corpus = corpus

        def get_scores(self, query):
            query_terms = set(query)
            return [float(sum(tokens.count(term) for term in query_terms)) for tokens in self.corpus]
from tqdm import tqdm


class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from a list of text chunks."""
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """Create and save BM25 indices for all reports."""

        output_dir.mkdir(parents=True, exist_ok=True)

        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(
            all_report_paths,
            desc="Processing reports for BM25"
        ):
            # Load report
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)

            # Extract chunks
            text_chunks = [
                str(chunk.get("text", ""))
                for chunk in report_data.get("content", {}).get("chunks", [])
            ]

            # Create BM25 index
            bm25_index = self.create_bm25_index(text_chunks)

            # Save index
            sha1_name = report_data.get("metainfo", {}).get("sha1_name", report_path.stem)
            output_file = output_dir / f"{sha1_name}.pkl"

            with open(output_file, "wb") as f:
                pickle.dump(bm25_index, f)

        print(f"Processed {len(all_report_paths)} reports")
