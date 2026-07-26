"""Small interfaces shared by VOX entities."""
from abc import ABC, abstractmethod

__all__ = [
    "VoxEntity",
    "Validateable",
]


class VoxEntity(ABC):
    """An abstract base class for objects that directly represent an entity in VOX file format."""

    @abstractmethod
    def to_vox_string(self) -> str:
        """Convert the object to its string representation in VOX file format."""
        raise NotImplementedError


class Validateable(ABC):
    """An abstract base class for classes that require validation."""

    @abstractmethod
    def validate(self):
        """
        Perform validation on the object.

        :raises ValueError: if any of the input is invalid.
        """
        raise NotImplementedError
