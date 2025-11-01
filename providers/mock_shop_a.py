# providers/mock_shop_a.py
import asyncio, random
from typing import List
from models import CompareQuery, PriceItem

class MockShopA:
    name = "MockShopA"

    async def search(self, q: CompareQuery, limit: int = 10) -> List[PriceItem]:
        await asyncio.sleep(0.05)  # 模拟网络
        base = 1880 if "iphone" in q.text.lower() else 120.0
        items = []
        for _ in range(min(3, limit)):
            price = round(base + random.uniform(-25, 25), 2)
            items.append(PriceItem(
                title=f"{q.text} - Retail Pack A",
                brand="Apple" if "iphone" in q.text.lower() else None,
                model="MTPV3" if "iphone" in q.text.lower() else None,
                variant="256GB Blue" if "iphone" in q.text.lower() else None,
                price=price,
                currency=q.currency,
                shipping_cost=0.0,
                tax_cost=0.0,
                seller="MockShop A",
                seller_rating=4.6,
                condition="new",
                url="https://shopA.example/item/123",
                source=self.name
            ))
        return items
