from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
#from sentence_transformers import SentenceTransformer

# Modelos para NER y POS
model_ner = "xlm-roberta-large-finetuned-conll03-english"
model_pos = "vblagoje/bert-english-uncased-finetuned-pos"

# Cargar los tokenizadores y modelos
tokenizer_ner = AutoTokenizer.from_pretrained(model_ner)
model_ner = AutoModelForTokenClassification.from_pretrained(model_ner)

tokenizer_pos = AutoTokenizer.from_pretrained(model_pos)
model_pos = AutoModelForTokenClassification.from_pretrained(model_pos)

# Modelo de similaridad
#simil_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Crear pipelines para NER y POS
ner_pipeline = pipeline("ner", model=model_ner, tokenizer=tokenizer_ner, aggregation_strategy="simple")
pos_pipeline = pipeline("token-classification", model=model_pos, tokenizer=tokenizer_pos)