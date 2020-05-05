import torch
import torch.nn as nn

from models.modules.attention import Attention


class TextSchemaEncoder(nn.Module):
    """Encodes the text and schema using co-attention."""

    def __init__(
            self,
            attention_key_size,
            encoder_num_layers=1,
            dropout=0):
        super().__init__()

        dropout = dropout if encoder_num_layers > 1 else 0

        # Self-attention
        self.schema2schema_attention = Attention(attention_key_size, attention_key_size)

        # Text-level attention
        # self.text_attention = Attention(attention_key_size)

        # Use attention module between input_hidden_states and schema_states
        self.text2schema_attention = Attention(attention_key_size, attention_key_size)
        self.schema2text_attention = Attention(attention_key_size, attention_key_size)

        # The second layer bi-lstms
        self.schema_encoder_2 = nn.LSTM(
            attention_key_size*2, # Concatenation
            attention_key_size//2,
            encoder_num_layers,
            dropout=dropout,
            bidirectional=True)
            
        self.text_encoder_2 = nn.LSTM(
            attention_key_size*2,
            attention_key_size//2,
            encoder_num_layers,
            dropout=dropout,
            bidirectional=True)

    def forward(
            self,
            schema_embs,
            text_embs):
        """Generates column head embeddings and text token embeddings.
        
        Args:
            schema_embs: schema_len x batch_size x emb_dim
            text_embs: text_len x batch_size x emb_dim

        Returns:
            schema_embs
            text_embs
            final_text_state: (hn, cn)
        """
        # len x batch_size x attention_dim
        schema_embs = self.schema2schema_attention(schema_embs, schema_embs)
        schema_attention = self.text2schema_attention(schema_embs, text_embs)
        text_attention = self.schema2text_attention(text_embs, schema_embs)

        # len x batch_size x (emb_dim + attention_dim)
        schema_embs = torch.cat((schema_embs, schema_attention), -1)
        text_embs = torch.cat((text_embs, text_attention), -1)

        # len x batch_size x 
        schema_encodings, _ = self.schema_encoder_2(schema_embs)
        text_token_encodings, final_text_state = self.text_encoder_2(text_embs)

        return schema_encodings, text_token_encodings, final_text_state
