import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .attention import Attention


class TextSchemaEncoder(nn.Module):
    """ Encodes the text and schema using co-attention. """

    def __init__(self, params):
        super().__init__()

        # unpack params
        self.schema_encoder_num_layer = 1
        self.schema_encoder_input_size = params.schema_encoder_input_size
        self.schema_encoder_state_size = params.schema_encoder_state_size
        self.schema_attention_key_size = params.schema_attention_key_size
        self.use_text_schema_attention = params.use_text_schema_attention
        self.encoder_num_layer = 0
        self.encoder_state_size = params.encoder_state_size
        self.text_attention_key_size = 0

        # create the schema encoder
        self.schema_encoder = Encoder(self.schema_encoder_num_layer, self.schema_encoder_input_size, self.schema_encoder_state_size)

        # self-attention
        self.schema2schema_attention = Attention(self.schema_attention_key_size, self.schema_attention_key_size, self.schema_attention_key_size)

        # text-level attention
        self.text_attention = Attention(self.encoder_state_size, self.encoder_state_size, self.encoder_state_size)

        # use attention module between input_hidden_states and schema_states
        #   schema_states: self.schema_attention_key_size x len(schema)
        #   input_hidden_states: self.text_attention_key_size x len(input)
        if self.use_text_schema_attention:
            self.text2schema_attention = Attention(self.schema_attention_key_size, self.text_attention_key_size, self.text_attention_key_size)
            self.schema2text_attention = Attention(self.text_attention_key_size, self.schema_attention_key_size, self.schema_attention_key_size)

        # concatenation
        self.schema_attention_key_size = self.schema_attention_key_size + self.text_attention_key_size
        self.text_attention_key_size = self.schema_attention_key_size + self.text_attention_key_size

        self.schema_encoder_2 = Encoder(self.schema_encoder_num_layer, self.schema_attention_key_size, self.schema_attention_key_size)
        self.text_encoder_2 = Encoder(self.encoder_num_layers, self.text_attention_key_size, self.text_attention_key_size)

    def predict_turn(
            self,
            utterance_final_state,
            input_hidden_states,
            schema_states,
            max_generation_length,
            gold_query=None,
            snippets=None,
            input_sequence=None,
            previous_queries=None,
            previous_query_states=None,
            input_schema=None,
            feed_gold_tokens=False,
            training=False):
        