import spacy
python -m spacy download en_core_web_sm


# class tests in tests

class SpacyNER():
    def __init__(self, text, lib="en_core_web_sm"):
        self.lib = lib
        self.text = text
        self._build_model()
        self.load_text()

    def _build_model(self):
        # modèle nlp est une instance Spacy
        self.nlp = spacy.load(self.lib)

    def load_text(self):
        self.doc = self.nlp(self.text)

    def print_ner_analysis(self):
        # Print the named entities recognized in the text
        print("Entities in the text:")
        for ent in self.doc.ents:
            print(f"{ent.text} - {ent.label_}")

