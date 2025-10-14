from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="CRUD Demo with FastAPI")

# --- Model ---
class Item(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float

# --- Fake DB ---
items_db: List[Item] = []

# --- CREATE ---
@app.post("/items/", response_model=Item)
def create_item(item: Item):
    for existing_item in items_db:
        if existing_item.id == item.id:
            raise HTTPException(status_code=400, detail="Item ID already exists")
    items_db.append(item)
    return item

# --- READ ALL ---
@app.get("/items/", response_model=List[Item])
def get_items():
    return items_db

# --- READ ONE ---
@app.get("/items/{item_id}", response_model=Item)
def get_item(item_id: int):
    for item in items_db:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")

# --- UPDATE ---
@app.put("/items/{item_id}", response_model=Item)
def update_item(item_id: int, updated_item: Item):
    for idx, item in enumerate(items_db):
        if item.id == item_id:
            items_db[idx] = updated_item
            return updated_item
    raise HTTPException(status_code=404, detail="Item not found")

# --- DELETE ---
@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    for idx, item in enumerate(items_db):
        if item.id == item_id:
            del items_db[idx]
            return {"message": f"Item {item_id} deleted successfully"}
    raise HTTPException(status_code=404, detail="Item not found")
