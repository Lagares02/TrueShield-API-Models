from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline

# Modelos para NER y POS
model_ner = "xlm-roberta-large-finetuned-conll03-english"
model_pos = "vblagoje/bert-english-uncased-finetuned-pos"

# Nombre del modelo de inferencia preentrenado a utilizar
inference_model = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

# Cargar los tokenizadores y modelos
tokenizer_ner = AutoTokenizer.from_pretrained(model_ner)
model_ner = AutoModelForTokenClassification.from_pretrained(model_ner)

tokenizer_pos = AutoTokenizer.from_pretrained(model_pos)
model_pos = AutoModelForTokenClassification.from_pretrained(model_pos)

# Modelos de traducción
tokenizer_en_es = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
model_en_es = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")

tokenizer_es_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
model_es_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")

# Cargar el tokenizador y el modelo preentrenado
tokenizer_inference = AutoTokenizer.from_pretrained(inference_model)
model_inference = AutoModelForSequenceClassification.from_pretrained(inference_model)

# Crear pipelines para NER y POS
ner_pipeline = pipeline("ner", model=model_ner, tokenizer=tokenizer_ner, aggregation_strategy="simple")
pos_pipeline = pipeline("token-classification", model=model_pos, tokenizer=tokenizer_pos)
translater_es_en_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
translater_en_es_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
inference_pipeline = pipeline("text-classification", model=inference_model)