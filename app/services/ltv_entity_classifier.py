class LTV_Entitye_Clasifier_Local():
    """
    ### Modelo de clasificación de entidades utilizando modelos locales
    """

    # List of non-relevant Part-of-Speech (PoS) entities
    PoS_entities_not_desirable = [
        "PUNCT", "CCONJ", "ADP", "DET", "PART", "AUX", "SCONJ", "SYM", "INTJ"
    ]

    def __init__(self, ner_pipeline, pos_pipeline):
        """
        Inicializa el clasificador con los pipelines locales para NER y PoS.
        """
        self.ner_pipeline = ner_pipeline
        self.pos_pipeline = pos_pipeline

    def token_classification(self, text, type_clasifier, result):
        """
        Método para clasificar tokens (NER o PoS)
        """
        if type_clasifier == "NER":
            entities = self.ner_pipeline(text)
        elif type_clasifier == "PoS":
            entities = self.pos_pipeline(text)
        else:
            raise ValueError("Invalid type_entity. Use 'NER' or 'PoS'.")

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
        else:
            for item in entities:
                if item["entity"] in self.PoS_entities_not_desirable:
                    continue
                result["entityes_PoS"].append({
                    "entitye": item['entity'],
                    "word": item['word']
                })

    def get(self, sentence):
        """
        Método para procesamiento de texto y obtención de entidades
        """
        result = {
            "text": sentence,
            "entityes_NER": [],
            "entityes_PoS": [],
            "key_words": []
        }

        self.token_classification(sentence, "NER", result)
        self.token_classification(sentence, "PoS", result)

        words_of_PoS = result["entityes_PoS"]
        words_of_NER = [kw["word"].lower() for kw in result["entityes_NER"]]

        for i in range(len(words_of_PoS)):

            # Validación de entidad individual
            noun = words_of_PoS[i]["entitye"]
            if noun == "NOUN" and len(words_of_PoS[i]["word"].split(' ')) >= 2:
                eng_word = words_of_PoS[i]["word"]
                kw = {"entitye": "MISC", "word": eng_word}
                result['key_words'].append(kw)

            propm = words_of_PoS[i]["entitye"]
            if propm == "PROPN":
                value = words_of_PoS[i]["word"].lower()
                if value in words_of_NER:
                    continue

            # Combinaciones de entidades
            if i + 2 < len(words_of_PoS):
                adj = words_of_PoS[i]["entitye"]
                noun = words_of_PoS[i+1]["entitye"]
                if adj == "ADJ" and noun == "NOUN":
                    eng_word = f"{words_of_PoS[i]['word']} {words_of_PoS[i+1]['word']}"
                    kw = {"entitye": "MISC", "word": eng_word}
                    result['key_words'].append(kw)

            if i + 1 < len(words_of_PoS):
                first_noun = words_of_PoS[i]["entitye"]
                second_noun = words_of_PoS[i+1]["entitye"]
                if first_noun == "NOUN" and second_noun == "NOUN":
                    eng_word = f"{words_of_PoS[i]['word']} {words_of_PoS[i+1]['word']}"
                    kw = {"entitye": "MISC", "word": eng_word}
                    result['key_words'].append(kw)

        response = {
            "prompt": result["text"],
            "ner": result["entityes_NER"],
            "pos": result["entityes_PoS"],
            "keywords": list(set([kw["word"] for kw in result["key_words"]]))
        }
        return response