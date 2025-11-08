from fastapi import APIRouter
from typing import Dict

router = APIRouter(prefix="/cart", tags=["Cart"])

cart = {}

@router.post("/add/{product_id}")
def add_to_cart(product_id: int, quantity: int = 1):
    cart[product_id] = cart.get(product_id, 0) + quantity
    return {"message": f"Added product {product_id} x{quantity} to cart", "cart": cart}

@router.post("/remove/{product_id}")
def remove_from_cart(product_id: int):
    if product_id in cart:
        del cart[product_id]
        return {"message": f"Removed product {product_id} from cart", "cart": cart}
    return {"error": "Product not in cart"}

@router.get("/")
def view_cart():
    return {"cart": cart}
