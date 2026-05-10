"""
Claim extraction module for the Disinformation Detection System.

Exports ClaimExtractor and the Claim dataclass.
"""

from claim_extraction.extractor import Claim, ClaimExtractor
from claim_extraction.ner_module import NERModule, NamedEntity

__all__ = ["ClaimExtractor", "Claim", "NERModule", "NamedEntity"]
