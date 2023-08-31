import torch
from transformers import AutoTokenizer, AutoModel
LLM_NAME = "distilbert-base-uncased"

use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = "cpu"
torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
llm_model = AutoModel.from_pretrained(LLM_NAME).to(device)

def get_text_embedding(text):
    with torch.no_grad():
        encoded_input = tokenizer(text, return_tensors='pt', padding=True)
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)
        lang_embedding = llm_model(input_ids, attention_mask=attention_mask).last_hidden_state
        # check this
        # lang_embedding = lang_embedding.mean(1)
        return lang_embedding

str1 = "hey"
str2 = "hello"
str3 = "go away. Dont come back ever again"

emb1 = get_text_embedding(str1)
emb2 = get_text_embedding(str2)
emb3 = get_text_embedding(str3)

# print("s1, s2: ", torch.cdist(emb1, emb2, p=2))
# print("s2, s3: ", torch.cdist(emb2, emb3, p=2))

print("emb1: ", emb1.shape)
print("emb2: ", emb2.shape)
print("emb3: ", emb3.shape)
