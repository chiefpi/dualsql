import torch
import torch.nn as nn

from attention import Attention


class TextSchemaEncoder(nn.Module):
    """Encodes the text and schema using co-attention."""

    def __init__(
            self,
            schema_encoder_input_size,
            schema_encoder_state_size,
            schema_encoder_num_layers,
            schema_attention_key_size,
            encoder_input_size,
            encoder_state_size,
            encoder_num_layers,
            text_attention_key_size,
            dropout=0):
        super().__init__()

        # Encoders
        self.text_encoder = nn.LSTM(
            encoder_input_size,
            encoder_state_size,
            encoder_num_layers,
            dropout=dropout,
            bidirectional=True)

        self.schema_encoder = nn.LSTM(
            schema_encoder_input_size,
            schema_encoder_state_size,
            schema_encoder_num_layers,
            dropout=dropout,
            bidirectional=True)

        # Self-attention
        self.schema2schema_attention = Attention(
            schema_attention_key_size,
            schema_attention_key_size,
            schema_attention_key_size)

        # Text-level attention
        self.text_attention = Attention(
            encoder_state_size,
            encoder_state_size,
            encoder_state_size)

        # Use attention module between input_hidden_states and schema_states
        # schema_states: self.schema_attention_key_size x len(schema)
        # input_hidden_states: self.text_attention_key_size x len(input)
        if self.use_text_schema_attention:
            self.text2schema_attention = Attention(
                schema_attention_key_size,
                text_attention_key_size,
                text_attention_key_size)
            self.schema2text_attention = Attention(
                text_attention_key_size,
                schema_attention_key_size,
                schema_attention_key_size)

        # Concatenation
        schema_attention_key_size = text_attention_key_size = \
            schema_attention_key_size + text_attention_key_size

        # The second layer bi-lstms
        self.schema_encoder_2 = nn.LSTM(
            schema_attention_key_size,
            schema_attention_key_size,
            schema_encoder_num_layers,
            dropout=dropout,
            bidirectional=True)
            
        self.text_encoder_2 = nn.LSTM(
            text_attention_key_size,
            text_attention_key_size,
            encoder_num_layers,
            dropout=dropout,
            bidirectional=True)

    def forward(
            self,
            schema_states,
            input_states):
        """Generates the column head encodings and text token encodings.
        
        Args:
            schema_states (list of Tensor)
            input_states (list of Tensor)
        """

        schema_states = self.schema_encoder(schema_states)
        input_hidden_states = self.text_encoder(input_states)

        schema_attention = self.text2schema_attention(
            torch.stack(schema_states, dim=0),
            input_hidden_states).vector # input_value_size x len(schema)
        text_attention = self.schema2text_attention(
            torch.stack(input_hidden_states, dim=0),
            schema_states).vector # schema_value_size x len(input)

        if schema_attention.dim() == 1:
            schema_attention = schema_attention.unsqueeze(1)
        if text_attention.dim() == 1:
            text_attention = text_attention.unsqueeze(1)

        # (input_value_size+schema_value_size) x len(schema)
        new_schema_states = torch.cat([
            torch.stack(schema_states, dim=1),
            schema_attention], dim=0)
        schema_states = list(torch.split(
            new_schema_states,
            split_size_or_sections=1,
            dim=1))
        schema_states = [schema_state.squeeze() for schema_state in schema_states]

        new_input_hidden_states = torch.cat([
            torch.stack(input_hidden_states, dim=1),
            text_attention], dim=0) # (input_value_size+schema_value_size) x len(input)
        input_hidden_states = list(torch.split(
            new_input_hidden_states,
            split_size_or_sections=1,
            dim=1))
        input_hidden_states = [input_hidden_state.squeeze() for input_hidden_state in input_hidden_states]

        # bi-lstm over schema_states and input_hidden_states
        # (embedder is an identity function)
        final_schema_state, schema_states = self.schema_encoder_2(
            schema_states)
        final_text_state, input_hidden_states = self.text_encoder_2(
            input_hidden_states)

        return final_schema_state, final_text_state
    
    def encode_schema_bow_simple(self, schema):
        schema_states = []
        for column_name in schema.column_names_embedder_input: # TODO: embedder input?
            # assert schema.in_vocab(column_name, surface_form=False)
            column_name_embeddings = [self.column_name_token_embedder(token) for token in column_name.split()]
            column_name_embeddings = torch.stack(column_name_embeddings, dim=0).mean(dim=0)
            schema_states.append(column_name_embeddings)
        return schema_states

    def encode_schema_self_attention(self, schema_states):
        schema_self_attention = self.schema2schema_attention(torch.stack(schema_states,dim=0), schema_states).vector
        if schema_self_attention.dim() == 1:
            schema_self_attention = schema_self_attention.unsqueeze(1)
        residual_schema_states = list(torch.split(schema_self_attention, split_size_or_sections=1, dim=1))
        residual_schema_states = [schema_state.squeeze() for schema_state in residual_schema_states]

        new_schema_states = [schema_state+residual_schema_state for schema_state, residual_schema_state in zip(schema_states, residual_schema_states)]

        return new_schema_states

    def encode_schema(self, input_schema, dropout=False):
      schema_states = []
      for column_name_embedder_input in input_schema.column_names_embedder_input:
        tokens = column_name_embedder_input.split()

        if dropout:
          final_schema_state_one, schema_states_one = self.schema_encoder(tokens, self.column_name_token_embedder, dropout_amount=self.dropout)
        else:
          final_schema_state_one, schema_states_one = self.schema_encoder(tokens, self.column_name_token_embedder)

        # final_schema_state_one: 1 means hidden_states instead of cell_memories, -1 means last layer
        schema_states.append(final_schema_state_one[1][-1])

      input_schema.set_column_name_embeddings(schema_states)

      # self-attention over schema_states
      if self.params.use_schema_self_attention:
        schema_states = self.encode_schema_self_attention(schema_states)

      return schema_states

    def get_bert_encoding(self, input_sequence, input_schema, discourse_state, dropout):
        text_states, schema_token_states = utils_bert.get_bert_encoding(self.bert_config, self.model_bert, self.tokenizer, input_sequence, input_schema, bert_input_version=self.params.bert_input_version, num_out_layers_n=1, num_out_layers_h=1)

        if self.params.discourse_level_lstm:
            text_token_embedder = lambda x: torch.cat([x, discourse_state], dim=0)
        else:
            text_token_embedder = lambda x: x

        if dropout:
            final_text_state, text_states = self.text_encoder(
                text_states,
                text_token_embedder,
                dropout_amount=self.dropout)
        else:
            final_text_state, text_states = self.text_encoder(
                text_states,
                text_token_embedder)

        schema_states = []
        for schema_token_states1 in schema_token_states:
            if dropout:
                final_schema_state_one, schema_states_one = self.schema_encoder(schema_token_states1, lambda x: x, dropout_amount=self.dropout)
            else:
                final_schema_state_one, schema_states_one = self.schema_encoder(schema_token_states1, lambda x: x)

            # final_schema_state_one: 1 means hidden_states instead of cell_memories, -1 means last layer
            schema_states.append(final_schema_state_one[1][-1])

        input_schema.set_column_name_embeddings(schema_states)

        # self-attention over schema_states
        if self.params.use_schema_self_attention:
            schema_states = self.encode_schema_self_attention(schema_states)

        return final_text_state, text_states, schema_states

    def get_query_token_embedding(self, output_token, input_schema):
        if input_schema:
            if not (self.output_embedder.in_vocabulary(output_token) or input_schema.in_vocabulary(output_token, surface_form=True)):
                output_token = 'value'
            if self.output_embedder.in_vocabulary(output_token):
                output_token_embedding = self.output_embedder(output_token)
            else:
                output_token_embedding = input_schema.column_name_embedder(output_token, surface_form=True)
        else:
            output_token_embedding = self.output_embedder(output_token)
        return output_token_embedding

    def get_text_attention(self, final_text_states_c, final_text_states_h, final_text_state, num_texts_to_keep):
        # self-attention between text_states            
        final_text_states_c.append(final_text_state[0][0])
        final_text_states_h.append(final_text_state[1][0])
        final_text_states_c = final_text_states_c[-num_texts_to_keep:]
        final_text_states_h = final_text_states_h[-num_texts_to_keep:]

        attention_result = self.text_attention_module(final_text_states_c[-1], final_text_states_c)
        final_text_state_attention_c = final_text_states_c[-1] + attention_result.vector.squeeze()

        attention_result = self.text_attention_module(final_text_states_h[-1], final_text_states_h)
        final_text_state_attention_h = final_text_states_h[-1] + attention_result.vector.squeeze()

        final_text_state = ([final_text_state_attention_c],[final_text_state_attention_h])

        return final_text_states_c, final_text_states_h, final_text_state