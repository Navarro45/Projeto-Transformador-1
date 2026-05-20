"""
Mantido por compatibilidade: o fluxo antigo chamava ``preprocess`` e depois ``split``.
A materialização completa está em ``preprocess.run``; aqui apenas reexportamos.
"""

from shared.pipelines.preprocess import run

__all__ = ["run"]
