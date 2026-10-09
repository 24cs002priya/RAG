'''For your consulting RAG, it should help with things such as:

User Query
     ↓
Understand query
     ↓
Identify relevant filters
     ↓
Prepare retrieval query

For example:

"Show me the 2024 market growth from automotive reports."

Potentially identify:

Year = 2024
Document Type = Research Report
Industry = Automotive
Concept applied

Query Analysis / Query Processing

Result

The retrieval layer receives a cleaner query and, where possible, structured filters.'''

import re
from typing import Any, Dict, Optional

from .metadata_service import normalize_metadata


class QueryProcessor:
    """Prepare a retrieval query and extract simple structured metadata filters."""

    YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2}|21\d{2})\b")

    def process(self, query: str, explicit_filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        filters = normalize_metadata(explicit_filters)
        year = self.YEAR_RE.search(query)
        if year and "year" not in filters:
            filters["year"] = int(year.group(1))

        retrieval_query = self.YEAR_RE.sub("", query.strip(), count=1) if year and "year" not in filters else query.strip()
        retrieval_query = " ".join(retrieval_query.split())
        return {
            "query": query.strip(),
            "retrieval_query": retrieval_query.strip() or query.strip(),
            "filters": filters,
        }
