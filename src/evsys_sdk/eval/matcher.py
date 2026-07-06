"""Alias-aware slug matching for tool-detection evals.

Wraps verified_aliases.json and an optional secondary-aliases file so a
predicted slug counts as correct if it equals the expected slug OR any
of the expected slug's verified aliases.

Bidirectional by default: an alias map ``{old: new}`` is treated as a
symmetric equivalence. This matters when the eval set has been renamed
to canonical/new slugs but the model under test was trained on the
old slugs (or vice versa) — both names should count as the same answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AliasMatcher:
    """Looks up whether ``predicted`` is an accepted answer for ``expected``.

    Inputs:
      * primary_aliases — flat map {slug_a: slug_b}. Treated as a symmetric
        equivalence: ``slug_a`` and ``slug_b`` are accepted interchangeably.
      * secondary_aliases — {slug_a: [slug_b, slug_c, ...]} for cases where
        multiple slugs are valid replacements (e.g. GOOGLESHEETS_QUERY_TABLE
        → {BATCH_GET, LOOKUP_SPREADSHEET_ROW, VALUES_GET}). Also symmetric.
      * bidirectional — set to False to disable reverse lookup (forward
        match only, like the original semantics).
    """

    primary_aliases: dict[str, str] = field(default_factory=dict)
    secondary_aliases: dict[str, list[str]] = field(default_factory=dict)
    bidirectional: bool = True

    # Pre-computed equivalence classes for O(1) lookup.
    _equivalence: dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._build_equivalence()

    def _build_equivalence(self) -> None:
        """Build {slug: {all_equivalents}} via union-find over alias pairs."""
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for a, b in self.primary_aliases.items():
            if self.bidirectional:
                union(a, b)
            else:
                # Asymmetric: a -> b only; record a tagged class for a.
                find(a)
                find(b)
                parent.setdefault(a, find(b))

        for a, bs in self.secondary_aliases.items():
            for b in bs:
                if self.bidirectional:
                    union(a, b)
                else:
                    find(a)
                    find(b)
                    parent.setdefault(a, find(b))

        classes: dict[str, set[str]] = {}
        for slug in list(parent.keys()):
            root = find(slug)
            classes.setdefault(root, set()).add(slug)

        # Map each slug to its full equivalence class.
        eq: dict[str, set[str]] = {}
        for root, members in classes.items():
            for m in members:
                eq[m] = members
        self._equivalence = eq

    @classmethod
    def from_files(
        cls,
        primary_path: str | Path,
        secondary_path: str | Path | None = None,
        *,
        bidirectional: bool = True,
    ) -> AliasMatcher:
        primary = json.loads(Path(primary_path).read_text()) if primary_path else {}
        secondary = (
            json.loads(Path(secondary_path).read_text())
            if secondary_path and Path(secondary_path).exists()
            else {}
        )
        return cls(primary_aliases=primary, secondary_aliases=secondary, bidirectional=bidirectional)

    def accepted_slugs(self, expected: str) -> list[str]:
        """All slugs that count as a correct prediction for ``expected``."""
        if expected in self._equivalence:
            # Preserve a stable order: expected first, then sorted rest.
            others = sorted(s for s in self._equivalence[expected] if s != expected)
            return [expected, *others]
        return [expected]

    def matches(self, expected: str, predicted: str) -> bool:
        if not expected or not predicted:
            return False
        if expected == predicted:
            return True
        cls_a = self._equivalence.get(expected)
        return bool(cls_a) and predicted in cls_a

    def found_in(self, expected: str, returned: list[str]) -> bool:
        """True if any accepted slug appears in ``returned`` (a list of slugs)."""
        if not expected or not returned:
            return False
        accepted = self._equivalence.get(expected, {expected})
        return any(s in accepted for s in returned)
