from datetime import datetime
from models.models import tokenizer_inference, model_inference
import torch

class LTV_Entity_Classifier_Local():
    """
    ### Modelo de clasificación de entidades utilizando modelos locales
    """

    # Lista de entidades PoS no deseadas
    PoS_entities_not_desirable = ["PUNCT", "CCONJ", "ADP", "DET", "PART", "AUX", "SCONJ", "SYM", "INTJ"]

    def __init__(self, ner_pipeline, pos_pipeline, translater_es_en_pipeline, translater_en_es_pipeline):
        self.ner_pipeline = ner_pipeline
        self.pos_pipeline = pos_pipeline
        self.translater_es_en_pipeline = translater_es_en_pipeline
        self.translater_en_es_pipeline = translater_en_es_pipeline

    def token_classification(self, text, type_clasifier, result):
        """
        Clasifica tokens utilizando NER o PoS y los agrega al resultado.
        """
        if type_clasifier == "NER":
            entities = self.ner_pipeline(text)
        elif type_clasifier == "PoS":
            entities = self.pos_pipeline(text)
        else:
            raise ValueError("Invalid type_entity. Use 'NER' for Named Entity Recognition or 'PoS' for Part of Speech tagging.")

        if type_clasifier == "NER":
            for item in entities:
                result[f"entityes_{type_clasifier}"].append({
                    "entitye": item['entity_group'],
                    "word": item['word']
                })
                kw = {
                    "entitye": item['entity_group'],
                    "word": item['word'],
                }
                result["key_words"].append(kw)
        else:  # PoS
            for item in entities:
                if item["entity"] in self.PoS_entities_not_desirable:
                    continue
                result["entityes_PoS"].append({
                    "entitye": item['entity'],
                    "word": item['word']
                })

    def process_combinations(self, words_of_PoS, result):
        """
        Realiza validaciones y combinaciones de entidades PoS y las agrega a key_words.
        """
        words_of_NER = [kw["word"].lower() for kw in result["entityes_NER"]]

        for i in range(len(words_of_PoS)):
            noun = words_of_PoS[i]["entitye"]
            if noun == "NOUN" and len(words_of_PoS[i]["word"].split(' ')) >= 2:
                eng_word = words_of_PoS[i]["word"]
                kw = {"entitye": "MISC", "word": eng_word}
                result['key_words'].append(kw)

            propn = words_of_PoS[i]["entitye"]
            if propn == "PROPN" and words_of_PoS[i]["word"].lower() not in words_of_NER:
                continue

            if i + 2 < len(words_of_PoS):
                adj, noun = words_of_PoS[i]["entitye"], words_of_PoS[i+1]["entitye"]
                if adj == "ADJ" and noun == "NOUN":
                    eng_word = f"{words_of_PoS[i]['word']} {words_of_PoS[i+1]['word']}"
                    kw = {"entitye": "MISC", "word": eng_word}
                    result['key_words'].append(kw)

            if i + 1 < len(words_of_PoS):
                first_noun, second_noun = words_of_PoS[i]["entitye"], words_of_PoS[i+1]["entitye"]
                if first_noun == "NOUN" and second_noun == "NOUN":
                    eng_word = f"{words_of_PoS[i]['word']} {words_of_PoS[i+1]['word']}"
                    kw = {"entitye": "MISC", "word": eng_word}
                    result['key_words'].append(kw)

    def get(self, sentence):
        """
        Procesa el texto para obtener las entidades y retorna un diccionario estructurado.
        """
        result = {
            "text": sentence,
            "entityes_NER": [],
            "entityes_PoS": [],
            "key_words": []
        }

        # Traducción al inglés
        translated_text = self.translater_es_en_pipeline(sentence)[0]['translation_text']
        print("Translated text: ", translated_text)
        
        # Clasificación de tokens
        self.token_classification(translated_text, "NER", result)
        self.token_classification(translated_text, "PoS", result)
        
        print("NER: ", result["entityes_NER"])
        print("PoS: ", result["entityes_PoS"])

        # Procesamiento de combinaciones
        self.process_combinations(result["entityes_PoS"], result)

        # Estructura de keywords en ambos idiomas
        keywords = {
            "keywords_en": [kw["word"] for kw in result["key_words"]],
            "keywords_es": [self.translater_en_es_pipeline(kw["word"])[0]['translation_text'] for kw in result["key_words"]]
        }

        print("keywords_es: ", keywords["keywords_es"])
        print("keywords_en: ", keywords["keywords_en"])

        # Creación del response final
        response = {
            "prompt": translated_text,
            "temporality": datetime.now().strftime("%Y-%m-%d"),
            "location": [kw["word"] for kw in result["key_words"] if kw["entitye"] == "LOC"],
            "keywords": keywords,
            "subjects": [kw["word"] for kw in result["key_words"] if kw["entitye"] in ["PER", "ORG"]]
        }

        print("response: ", response)

        return response

def classify_text_relationship(premise, hypothesis):

    # Longitud máxima de las secuencias
    max_length = 280

    # Codificar la premisa y la hipótesis
    encoded_input = tokenizer_inference.encode_plus(
        premise, hypothesis,  # Premisa e hipótesis
        max_length=max_length,  # Longitud máxima
        return_token_type_ids=True,  # Devolver IDs de tipo de token
        truncation=True  # Truncar si excede el límite
    )

    # Preparar los tensores de entrada para el modelo
    input_ids = torch.tensor([encoded_input['input_ids']])
    token_type_ids = torch.tensor([encoded_input['token_type_ids']])
    attention_mask = torch.tensor([encoded_input['attention_mask']])

    # Obtener las predicciones del modelo
    with torch.no_grad():  # Desactivar el cálculo del gradiente (ahorra memoria)
        outputs = model_inference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    # Aplicar softmax para obtener las probabilidades de cada clase
    predicted_probabilities = torch.softmax(outputs[0], dim=1).squeeze().tolist()

    # Definir las categorías
    categories = ["affirmation", "assumption", "denial"]

    # Obtener la categoría con la mayor probabilidad
    max_prob_index = predicted_probabilities.index(max(predicted_probabilities))
    max_category = categories[max_prob_index]
    max_probability = predicted_probabilities[max_prob_index]

    print("inferencia: ", max_category)
    # Retornar la categoría con la mayor probabilidad y su valor
    return {
        "category": max_category,
        "probability": max_probability
    }