"""
Source policy for web evidence retrieval.

Loads the tiered allow-list and blocklist from ``configs/source_policy.yaml``
and decides, for any URL, whether it may supply evidence and how much it is
trusted. Matching is offline: ``tldextract`` uses its bundled public-suffix
snapshot and never touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

import tldextract
import yaml  # type: ignore[import-untyped]

from exceptions import RetrievalError
from logging_config import get_logger

if TYPE_CHECKING:
    from configs import Settings

__all__ = ["SourcePolicy", "SourceMatch", "PolicyMode"]

logger = get_logger(__name__)

PolicyMode = Literal["strict", "lenient", "off"]
_MODES = ("strict", "lenient", "off")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Offline extractor: bundled suffix list only, no disk cache.
_EXTRACT = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=())


@dataclass(frozen=True)
class _Entry:
    domain: str
    path: str
    tier: str
    trust: float

    @property
    def length(self) -> int:
        return len(self.domain) + len(self.path)


@dataclass(frozen=True)
class SourceMatch:
    """A tier match for a URL.

    Attributes:
        tier: Name of the matching tier.
        trust: The tier's trust value.
        entry: The matching domain entry (``domain`` or ``domain/path``).
    """

    tier: str
    trust: float
    entry: str


def _split_url(url: str) -> tuple[str, str]:
    """Return the lower-cased host (no port, no trailing dot) and the path."""
    parts = urlsplit(url if "//" in url else f"//{url}")
    host = (parts.hostname or "").lower().rstrip(".")
    return host, parts.path or "/"


def _host_matches(host: str, domain: str) -> bool:
    """True when *host* equals *domain* or is a subdomain of it."""
    return host == domain or host.endswith("." + domain)


def _path_matches(path: str, prefix: str) -> bool:
    """True when *path* starts with *prefix* at a path-segment boundary."""
    if not prefix:
        return True
    path = path.lower()
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def _parse_entry(raw: str) -> tuple[str, str]:
    """Split ``"reuters.com/fact-check"`` into ``("reuters.com", "/fact-check")``."""
    text = str(raw).strip().lower()
    if "://" in text:
        text = text.split("://", 1)[1]
    domain, _, path = text.partition("/")
    domain = domain.rstrip(".")
    if domain.startswith("www."):
        domain = domain[4:]
    path = f"/{path}".rstrip("/") if path else ""
    return domain, path


def publisher_for(url: str) -> str:
    """Registrable domain of *url* (``www.bbc.co.uk`` → ``bbc.co.uk``).

    Falls back to the bare host for hosts without a public suffix.
    """
    host, _ = _split_url(url)
    if not host:
        return ""
    result = _EXTRACT(host)
    registrable = getattr(result, "top_domain_under_public_suffix", None)
    if registrable is None:  # tldextract < 5.3
        registrable = result.registered_domain
    return registrable or host


class SourcePolicy:
    """Tiered allow-list and blocklist of evidence domains.

    Args:
        data: Parsed policy file (``version``, ``tiers``, ``blocklist``).
        mode: ``strict``, ``lenient`` or ``off``.
        unknown_trust: Trust given to unmatched URLs in ``lenient``/``off``.
        source: Where *data* came from, used in error messages.

    Raises:
        RetrievalError: If the policy is malformed or a trust value is
            outside [0, 1].
    """

    def __init__(
        self,
        data: dict[str, Any],
        mode: PolicyMode = "strict",
        unknown_trust: float = 0.3,
        source: str = "<policy>",
    ) -> None:
        if mode not in _MODES:
            raise RetrievalError(
                f"Unknown source policy mode {mode!r}; expected one of {list(_MODES)}."
            )
        self._mode: PolicyMode = mode
        self._unknown_trust = float(unknown_trust)
        self._source = source
        self._version = str(data.get("version", "")) if isinstance(data, dict) else ""
        self._entries, self._tier_order = self._parse_tiers(data)
        self._blocklist = self._parse_blocklist(data)

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        mode: PolicyMode = "strict",
        unknown_trust: float = 0.3,
    ) -> "SourcePolicy":
        """Load a policy file.

        Relative paths are resolved against the project root.

        Raises:
            RetrievalError: If the file is missing, unreadable or malformed.
        """
        p = Path(path)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        if not p.is_file():
            raise RetrievalError(f"Source policy file not found: '{p}'.")
        try:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as exc:
            raise RetrievalError(
                f"Source policy file '{p}' could not be parsed.", original_error=exc
            ) from exc
        policy = cls(data, mode=mode, unknown_trust=unknown_trust, source=str(p))
        logger.info(
            "Loaded source policy '%s' (version %s, %d entries, mode=%s).",
            p.name,
            policy.version,
            len(policy._entries),
            mode,
        )
        return policy

    @classmethod
    def from_settings(cls, settings: "Settings") -> "SourcePolicy":
        """Build the policy from ``settings.source_policy``."""
        cfg = settings.source_policy
        return cls.from_file(cfg.path, mode=cfg.mode, unknown_trust=cfg.unknown_trust)

    def _parse_tiers(
        self, data: Any
    ) -> tuple[list[_Entry], list[str]]:
        if not isinstance(data, dict) or not isinstance(data.get("tiers"), dict):
            raise RetrievalError(
                f"Source policy {self._source} must contain a 'tiers' mapping."
            )
        entries: list[_Entry] = []
        tiers: list[tuple[float, int, str]] = []
        for pos, (name, tier) in enumerate(data["tiers"].items()):
            if not isinstance(tier, dict):
                raise RetrievalError(
                    f"Source policy tier '{name}' must be a mapping with "
                    "'trust' and 'domains'."
                )
            try:
                trust = float(tier.get("trust"))  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise RetrievalError(
                    f"Source policy tier '{name}' has an invalid trust value "
                    f"{tier.get('trust')!r}.",
                    original_error=exc,
                ) from exc
            if not 0.0 <= trust <= 1.0:
                raise RetrievalError(
                    f"Source policy tier '{name}' has trust {trust}, which is "
                    "outside [0, 1]."
                )
            domains = tier.get("domains") or []
            if not isinstance(domains, list):
                raise RetrievalError(
                    f"Source policy tier '{name}' must list its domains."
                )
            for raw in domains:
                domain, path = _parse_entry(raw)
                if domain:
                    entries.append(_Entry(domain, path, str(name), trust))
            tiers.append((trust, pos, str(name)))
        tiers.sort(key=lambda t: (-t[0], t[1]))
        return entries, [name for _, _, name in tiers]

    def _parse_blocklist(self, data: dict[str, Any]) -> list[tuple[str, str]]:
        raw = data.get("blocklist") or []
        if not isinstance(raw, list):
            raise RetrievalError(
                f"Source policy {self._source}: 'blocklist' must be a list."
            )
        return [e for e in (_parse_entry(item) for item in raw) if e[0]]

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def mode(self) -> PolicyMode:
        """The filtering mode."""
        return self._mode

    @property
    def version(self) -> str:
        """The policy file's ``version`` value."""
        return self._version

    @property
    def unknown_trust(self) -> float:
        """Trust given to unmatched URLs in ``lenient`` and ``off`` modes."""
        return self._unknown_trust

    # ── matching ────────────────────────────────────────────────────────────

    def match(self, url: str) -> SourceMatch | None:
        """Return the tier match for *url*, or None when no entry matches.

        The blocklist and the mode are not considered here.
        """
        host, path = _split_url(url)
        if not host:
            return None
        best: _Entry | None = None
        for entry in self._entries:
            if _host_matches(host, entry.domain) and _path_matches(path, entry.path):
                if best is None or entry.length > best.length:
                    best = entry
        if best is None:
            return None
        return SourceMatch(
            tier=best.tier, trust=best.trust, entry=best.domain + best.path
        )

    def is_blocked(self, url: str) -> bool:
        """True when *url* matches a blocklist entry."""
        host, path = _split_url(url)
        return any(
            _host_matches(host, domain) and _path_matches(path, prefix)
            for domain, prefix in self._blocklist
        )

    def allows(self, url: str) -> bool:
        """Whether evidence from *url* may be used under the current mode."""
        if self._mode == "off":
            return True
        if self.is_blocked(url):
            return False
        if self._mode == "lenient":
            return True
        return self.match(url) is not None

    def tier_for(self, url: str) -> str:
        """Tier name for *url* ("" when unmatched or blocklisted)."""
        if self.is_blocked(url):
            return ""
        m = self.match(url)
        return m.tier if m else ""

    def trust_for(self, url: str) -> float:
        """Trust value for *url*.

        Matched URLs get their tier's trust; unmatched (and, in ``off`` mode,
        blocklisted) URLs get ``unknown_trust``.
        """
        if self.is_blocked(url):
            return self._unknown_trust
        m = self.match(url)
        return m.trust if m else self._unknown_trust

    # ── search restriction ──────────────────────────────────────────────────

    def site_groups(self, group_size: int) -> list[list[str]]:
        """Allow-listed domains for ``site:`` operators, chunked into groups.

        Domains are ordered by tier (highest trust first, then file order),
        then by file order within a tier. Path entries are reduced to their
        domain, duplicates are removed, and blocklisted domains are skipped.

        Args:
            group_size: Maximum number of domains per group (>= 1).

        Returns:
            List of domain groups; empty when the allow-list is empty.
        """
        if group_size < 1:
            raise ValueError("group_size must be >= 1")
        rank = {name: i for i, name in enumerate(self._tier_order)}
        ordered = sorted(
            enumerate(self._entries), key=lambda ie: (rank[ie[1].tier], ie[0])
        )
        blocked = {domain for domain, prefix in self._blocklist if not prefix}
        domains: list[str] = []
        seen: set[str] = set()
        for _, entry in ordered:
            if entry.domain in seen or entry.domain in blocked:
                continue
            seen.add(entry.domain)
            domains.append(entry.domain)
        return [
            domains[i : i + group_size] for i in range(0, len(domains), group_size)
        ]
