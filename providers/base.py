# providers/base.py
from typing import List
from abc import ABC, abstractmethod
from models import PriceItem, CompareQuery

class ProviderBase(ABC):
    name: str = "base"

    @abstractmethod
    async def search(self, q: CompareQuery, limit: int = 10) -> List[PriceItem]:
        ...
