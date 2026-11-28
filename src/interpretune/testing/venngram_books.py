"""Address books built around one treatment, for the tests that apply a venngram treatment.

Venngram reads a treatment only from the address book it treats (the book's ``treatments`` section): the standalone
``kind: recipe`` document is retired. A test usually states one treatment and the sets it targets, so
:func:`treatment_book` takes those two parts and returns the book, stamped with the lowest schema that holds it.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: The keys of a treatment spec that are not part of the treatment's body.
_SPEC_KEYS = frozenset({"name", "book", "kind", "schema_version"})


def treatment_book(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The address book a treatment spec describes: ``spec["book"]`` with ``spec``'s treatment under ``treatments``.

    Args:
        spec: The treatment's ``name``, its ``book`` (the book's own keys, without ``treatments``) and the treatment's
            body (``model``, ``operands``, ``stages``, ``weights`` and the rest), all at the top level. A ``kind`` or
            ``schema_version`` key is ignored.

    Returns:
        A book mapping ``venngram.book.AddressBook.from_mapping`` reads, its ``schema_version`` the lowest that holds
        it (``AddressBook.needed_schema``).
    """
    from venngram.book import AddressBook

    body = {k: v for k, v in spec.items() if k not in _SPEC_KEYS}
    book = {k: v for k, v in spec["book"].items() if k != "schema_version"}
    book["treatments"] = {**book.get("treatments", {}), spec["name"]: body}
    return {"schema_version": AddressBook.needed_schema(book), **book}
