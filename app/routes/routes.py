from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.logic import LTV_Entity_Classifier_Local, classify_text_relationship
from models.models import ner_pipeline, pos_pipeline, translater_es_en_pipeline, translater_en_es_pipeline
import requests

router = APIRouter()

# Instanciamos la clase clasificadora
entity_classifier = LTV_Entity_Classifier_Local(ner_pipeline, pos_pipeline, translater_es_en_pipeline, translater_en_es_pipeline)

@router.get("/")
def home():
    return {"message": "Corriendo exitosamente TrueShield-API-Models!"}

# Define un esquema para la petición
class PromptRequest(BaseModel):
    prompt: str

class InferenceRequest(BaseModel):
    premise: str
    hypothesis: str

# Definimos la ruta de la API para clasificar entidades
@router.post("/classify")
def classify(request: PromptRequest):
    try:
        print("prompt received:", request.prompt)
        result = entity_classifier.get(request.prompt)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/inference")
async def infer(request: InferenceRequest):
    try:
        # Traducir la entrada del español al inglés
        premise_en = translater_es_en_pipeline(request.premise)[0]['translation_text']
        hypothesis_en = translater_es_en_pipeline(request.hypothesis)[0]['translation_text']
        
        # Llamar a la función de clasificación de relación
        result = classify_text_relationship(premise_en, hypothesis_en)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))