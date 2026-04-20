from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel


class _BaseModel(BaseModel):
    @classmethod
    def field_names(cls) -> list[str]:
        return list(cls.model_fields)

    def __getitem__(self, key: str) -> Any:
        if key not in type(self).field_names():
            raise KeyError(f"Unknown field: {key}")
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in type(self).field_names():
            raise KeyError(f"Unknown field: {key}")
        setattr(self, key, value)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in type(self).field_names()

    def __iter__(self) -> Iterator[str]:
        return iter(type(self).field_names())

    def __len__(self) -> int:
        return len(type(self).field_names())

    def keys(self) -> list[str]:
        return type(self).field_names()

    def values(self) -> list[Any]:
        return [getattr(self, key) for key in type(self).field_names()]

    def items(self) -> list[tuple[str, Any]]:
        return [(key, getattr(self, key)) for key in type(self).field_names()]

    def get(self, key: str, default: Any = None) -> Any:
        if key not in type(self).field_names():
            return default
        value = getattr(self, key)
        return value if value is not None else default
