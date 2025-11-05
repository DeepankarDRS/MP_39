from fastapi import FastAPI
from routes import products, cart, orders

app = FastAPI(
    title="Zepto Prototype API",
    description="Lightweight FastAPI app simulating Zepto’s quick delivery backend",
    version="1.0"
)

# Register Routers
app.include_router(products.router)
app.include_router(cart.router)
app.include_router(orders.router)

@app.get("/")
def root():
    return {"message": "Welcome to Zepto FastAPI Prototype 🚀"}
