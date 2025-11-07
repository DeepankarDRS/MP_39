from fastapi import APIRouter

router = APIRouter(prefix="/products", tags=["Products"])

# Mock data
products = [
    {"id": 1, "name": "Bananas", "price": 30, "category": "Fruits"},
    {"id": 2, "name": "Milk", "price": 60, "category": "Dairy"},
    {"id": 3, "name": "Bread", "price": 40, "category": "Bakery"},
    {"id": 4, "name": "Tomatoes", "price": 25, "category": "Vegetables"},
]

@router.get("/")
def get_products():
    return {"products": products}

@router.get("/{product_id}")
def get_product(product_id: int):
    for p in products:
        if p["id"] == product_id:
            return p
    return {"error": "Product not found"}
