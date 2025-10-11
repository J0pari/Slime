"""Protocol definitions for slime mold components"""

from slime.proto.component import Component
from slime.proto.kernel import Kernel
from slime.proto.memory import Memory
from slime.proto.model import Organism, Pseudopod, Chemotaxis

__all__ = ["Component", "Kernel", "Memory", "Organism", "Pseudopod", "Chemotaxis"]
