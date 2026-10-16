"""Dependency-free smoke checks for citation reference numbering.

The module is loaded by file path to avoid importing the heavier evaluation
package and LangGraph dependencies. This checks the numeric-reference contract
only, not factual support or end-to-end response quality.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATORS = ROOT / "app" / "evaluation" / "evaluators.py"
spec = importlib.util.spec_from_file_location("_citation_contract_evaluators", EVALUATORS)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
citation_validity = module.citation_validity


class CitationReferenceContract(unittest.TestCase):
    def score(self, answer: str) -> float:
        return float(citation_validity({}, {"output": answer})["score"])

    def test_valid_inline_reference(self) -> None:
        self.assertEqual(self.score("Fact [1].\n\nSources:\n[1] doc.md"), 1.0)

    def test_bibliography_alone_is_not_inline_citation(self) -> None:
        self.assertEqual(self.score("Fact.\n\nSources:\n[1] doc.md"), 0.0)

    def test_no_sources_section_is_incomplete(self) -> None:
        self.assertEqual(self.score("Fact [1].\n\n[1] doc.md"), 0.5)

    def test_dangling_reference(self) -> None:
        self.assertEqual(self.score("Fact [2].\n\nSources:\n[1] doc.md"), 0.0)

    def test_partial_dangling_references(self) -> None:
        self.assertEqual(
            self.score("Fact [1] and another [2].\n\nSources:\n[1] doc.md"),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
