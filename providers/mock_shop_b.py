# providers/mock_shop_b.py
import asyncio, random
from typing import List
from models import CompareQuery, PriceItem

class MockShopB:
    name = "MockShopB"

    async def search(self, q: CompareQuery, limit: int = 10) -> List[PriceItem]:
        await asyncio.sleep(0.05)
        base = 1865 if "iphone" in q.text.lower() else 115.0
        items = []
        for _ in range(min(3, limit)):
            price = round(base + random.uniform(-20, 20), 2)
            items.append(PriceItem(
                title=f"{q.text} - Prime B Bundle",
                brand="Apple" if "iphone" in q.text.lower() else None,
                model="MTPV3" if "iphone" in q.text.lower() else None,
                variant="256GB Blue" if "iphone" in q.text.lower() else None,
                price=price,
                currency=q.currency,
                shipping_cost=12.0,
                tax_cost=0.0,
                seller="MockShop B",
                seller_rating=4.8,
                condition="new",
                url="https://shopB.example/item/xyz",
                source=self.name
            ))
        return items
