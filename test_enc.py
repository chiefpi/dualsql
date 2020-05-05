import torch

from models.modules.text_schema_encoder import TextSchemaEncoder

encoder = TextSchemaEncoder(
    50,
    dropout=0.5)

schema_embs = torch.randn(5,3,50)
text_embs = torch.randn(10,3,50)
schema_embs, text_embs, final_text_state = encoder(schema_embs, text_embs)
print(schema_embs.size())
print(text_embs.size())
print(final_text_state[0])
print(final_text_state[0].transpose(0, 1).contiguous())
