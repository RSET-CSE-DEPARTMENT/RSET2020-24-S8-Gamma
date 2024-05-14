import spacy

nlp = spacy.load('en_core_web_sm')

def replace_pronouns_with_coref(text):
    doc = nlp(text)
    resolved_text = doc._.coref_resolved
    return resolved_text

# Example usage
text = "He went to the store, but she stayed at home."
replaced_text = replace_pronouns_with_coref(text)
print(replaced_text)
