from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# -------------------------------
# Database Setup
# -------------------------------
SQLALCHEMY_DATABASE_URL = "sqlite:///./delivery.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------
# Models
# -------------------------------
class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    customer_name = Column(String, index=True)
    address = Column(String)
    item = Column(String)
    price = Column(Float)
    status = Column(String, default="Pending")

# -------------------------------
# Schemas
# -------------------------------
class OrderBase(BaseModel):
    customer_name: str
    address: str
    item: str
    price: float

class OrderCreate(OrderBase):
    pass

class OrderUpdate(BaseModel):
    status: str

class OrderOut(OrderBase):
    id: int
    status: str

    class Config:
        orm_mode = True

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="Delivery App 🚚")

# Create database tables
Base.metadata.create_all(bind=engine)

# -------------------------------
# Routes
# -------------------------------

@app.post("/orders/", response_model=OrderOut)
def create_order(order: OrderCreate, db: Session = Depends(get_db)):
    new_order = Order(**order.dict())
    db.add(new_order)
    db.commit()
    db.refresh(new_order)
    return new_order

@app.get("/orders/", response_model=List[OrderOut])
def get_orders(db: Session = Depends(get_db)):
    return db.query(Order).all()

@app.get("/orders/{order_id}", response_model=OrderOut)
def get_order(order_id: int, db: Session = Depends(get_db)):
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order

@app.put("/orders/{order_id}", response_model=OrderOut)
def update_order(order_id: int, update: OrderUpdate, db: Session = Depends(get_db)):
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    order.status = update.status
    db.commit()
    db.refresh(order)
    return order

@app.delete("/orders/{order_id}")
def delete_order(order_id: int, db: Session = Depends(get_db)):
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    db.delete(order)
    db.commit()
    return {"message": f"Order {order_id} deleted successfully."}
