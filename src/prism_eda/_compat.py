"""Small shims for the oldest Python version the package supports."""

from __future__ import annotations

import sys
from enum import Enum

__all__ = ["StrEnum"]

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """Backport of :class:`enum.StrEnum` for Python 3.10.

        A plain ``str, Enum`` mixin formats as ``ClassName.MEMBER``; the stdlib
        ``StrEnum`` formats as the value. Results are serialised and compared as
        strings everywhere, so the backport has to match the stdlib behaviour.
        """

        def __str__(self) -> str:
            return str(self.value)

        def __format__(self, format_spec: str) -> str:
            return format(str(self.value), format_spec)

        @staticmethod
        def _generate_next_value_(
            name: str, start: int, count: int, last_values: list[object]
        ) -> str:
            return name.lower()
