"""QA dataset handling for evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QAExample:
    """A single QA example for evaluation."""

    id: str
    question: str
    answer: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str | list[str] | dict[str, str]]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "relevant_doc_ids": self.relevant_doc_ids,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | list[str] | dict[str, str]]) -> QAExample:
        """Create from dictionary."""
        return cls(
            id=str(data["id"]),
            question=str(data["question"]),
            answer=str(data["answer"]),
            relevant_doc_ids=list(data.get("relevant_doc_ids", [])),
            relevant_chunk_ids=list(data.get("relevant_chunk_ids", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class QADataset:
    """A dataset of QA examples."""

    name: str
    description: str
    examples: list[QAExample]

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def to_dict(self) -> dict[str, str | list[dict[str, str | list[str] | dict[str, str]]]]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "examples": [e.to_dict() for e in self.examples],
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, str | list[dict[str, str | list[str] | dict[str, str]]]]
    ) -> QADataset:
        """Create from dictionary."""
        examples = [QAExample.from_dict(e) for e in data.get("examples", [])]  # type: ignore[arg-type]
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            examples=examples,
        )

    def save(self, path: Path | str) -> None:
        """Save dataset to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> QADataset:
        """Load dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


def load_qa_dataset(path: Path | str) -> QADataset:
    """Load a QA dataset from file."""
    return QADataset.load(path)
