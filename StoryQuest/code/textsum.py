import torch
#import spacy
#import neuralcoref
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "textsumm/textsumm"
tokenizer_path = "tokenizer/tokenizer"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

ARTICLE_TO_SUMMARIZE = '''Once upon a time, in a faraway kingdom, Prince set out on a journey. Prince encountered a wise old sage who gifted Prince a magical amulet. With the amulet in hand, Prince embarked on a quest to save the kingdom from darkness. Along the way, Prince faced many challenges and adversaries. Despite the dangers, Prince remained steadfast and courageous. Eventually, Prince reached the heart of the kingdom's troubles and used the power of the amulet to vanquish the evil. The kingdom rejoiced, celebrating Prince as a hero. From that day forth, Prince's name was honored and remembered throughout the land.'''

inputs = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors="pt")

# Global attention on the first token (cf. Beltagy et al. 2020)
global_attention_mask = torch.zeros_like(inputs)
global_attention_mask[:, 0] = 1

# Generate Summary
summary_ids = model.generate(inputs, global_attention_mask=global_attention_mask, num_beams=3, max_length=150)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

print(summary)