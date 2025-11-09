from fastapi import APIRouter
import random

router = APIRouter(prefix="/orders", tags=["Orders"])

orders = {}

@router.post("/place")
def place_order():
    if not orders:
        order_id = random.randint(1000, 9999)
        orders[order_id] = {"status": "Placed"}
        return {"message": "Order placed successfully", "order_id": order_id}
    else:
        order_id = list(orders.keys())[0]
        return {"message": "Existing order in progress", "order_id": order_id}

@router.get("/{order_id}")
def get_order_status(order_id: int):
    if order_id in orders:
        return {"order_id": order_id, "status": orders[order_id]["status"]}
    return {"error": "Order not found"}

@router.post("/{order_id}/update")
def update_order_status(order_id: int, status: str):
    if order_id in orders:
        orders[order_id]["status"] = status
        return {"message": f"Order {order_id} updated to {status}"}
    return {"error": "Order not found"}
