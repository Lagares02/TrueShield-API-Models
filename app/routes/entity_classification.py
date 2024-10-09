from fastapi import APIRouter
from pydantic import BaseModel
from services.ltv_entity_classifier import LTV_Entitye_Clasifier_Local
from models.transformers_model import ner_pipeline, pos_pipeline

router = APIRouter()

# Instanciamos la clase clasificadora
entity_classifier = LTV_Entitye_Clasifier_Local(ner_pipeline, pos_pipeline)

@router.get("/")
def home():
    return {"message": "Corriendo exitosamente TrueShield-API-Models!"}

# Define un esquema para la petición
class PromptRequest(BaseModel):
    prompt: str

# Definimos la ruta de la API para clasificar entidades
@router.post("/classify")
def classify(request: PromptRequest):
    result = entity_classifier.get(request.prompt)
    return result