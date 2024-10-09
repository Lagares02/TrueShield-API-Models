from fastapi import FastAPI
from routes.entity_classification import router

app = FastAPI()

# Incluir las rutas
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)