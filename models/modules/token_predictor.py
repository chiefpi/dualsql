"""Predicts a token."""

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_utils
from attention import Attention, AttentionResult


class PredictionInput(namedtuple(
        'PredictionInput',
        ('decoder_state', 
        'input_hidden_states',
        'input_sequence'))):
    """Contains input to a token predictor."""
    __slots__ = ()

class PredictionInputWithSchema(namedtuple(
        'PredictionInputWithSchema',
        ('decoder_state',
        'input_hidden_states',
        'schema_states',
        'input_sequence',
        'previous_queries',
        'previous_query_states',
        'input_schema'))):
    """Contains input to a token predictor."""
    __slots__ = ()

class TokenPrediction(namedtuple(
        'TokenPrediction',
        ('scores',
        'aligned_tokens',
        'utterance_attention_results',
        'schema_attention_results',
        'query_attention_results',
        'copy_switch',
        'query_scores',
        'query_tokens',
        'decoder_state'))):
    """Contains a token prediction."""
    __slots__ = ()

def score_schema_tokens(input_schema, schema_states, scorer):
    # schema_states: emd_dim x num_tokens
    scores = torch.t(torch.mm(torch.t(scorer), schema_states))   # num_tokens x 1
    if scores.size()[0] != len(input_schema):
        raise ValueError("Got " + str(scores.size()[0]) + " scores for " + str(len(input_schema)) + " schema tokens")
    return scores, input_schema.column_names_surface_form

def score_query_tokens(previous_query, previous_query_states, scorer):
    scores = torch.t(torch.mm(torch.t(scorer), previous_query_states))   # num_tokens x 1
    if scores.size()[0] != len(previous_query):
        raise ValueError("Got " + str(scores.size()[0]) + " scores for " + str(len(previous_query)) + " query tokens")
    return scores, previous_query

class TokenPredictor(nn.Module):
    """Predicts a token given a (decoder) state.

    Attributes:
        vocabulary (Vocabulary): A vocabulary object for the output.
        attention_module (Attention): An attention module.
        state_transformation_weights (dy.Parameters): Transforms the input state
            before predicting a token.
        vocabulary_weights (dy.Parameters): Final layer weights.
        vocabulary_biases (dy.Parameters): Final layer biases.
    """

    def __init__(self, params, vocabulary, attention_key_size):
        super().__init__()
        self.params = params
        self.vocabulary = vocabulary
        self.attention_module = Attention(params.decoder_state_size, attention_key_size, attention_key_size)
        self.state_transform_weights = torch_utils.add_params((params.decoder_state_size + attention_key_size, params.decoder_state_size), "weights-state-transform")
        self.vocabulary_weights = torch_utils.add_params((params.decoder_state_size, len(vocabulary)), "weights-vocabulary")
        self.vocabulary_biases = torch_utils.add_params(tuple([len(vocabulary)]), "biases-vocabulary")

    def _get_intermediate_state(self, state, dropout=0.):
        intermediate_state = torch.tanh(torch_utils.linear_layer(state, self.state_transform_weights))
        return F.dropout(intermediate_state, dropout)

    def _score_vocabulary_tokens(self, state):
        scores = torch.t(torch_utils.linear_layer(state, self.vocabulary_weights, self.vocabulary_biases))

        if scores.size()[0] != len(self.vocabulary.inorder_tokens):
            raise ValueError("Got " + str(scores.size()[0]) + " scores for " + str(len(self.vocabulary.inorder_tokens)) + " vocabulary items")

        return scores, self.vocabulary.inorder_tokens

    def forward(self, prediction_input, dropout=0.):
        decoder_state = prediction_input.decoder_state
        input_hidden_states = prediction_input.input_hidden_states

        attention_results = self.attention_module(decoder_state, input_hidden_states)

        state_and_attn = torch.cat([decoder_state, attention_results.vector], dim=0)

        intermediate_state = self._get_intermediate_state(state_and_attn, dropout=dropout)
        vocab_scores, vocab_tokens = self._score_vocabulary_tokens(intermediate_state)

        return TokenPrediction(vocab_scores, vocab_tokens, attention_results, decoder_state)


class SchemaTokenPredictor(TokenPredictor):

    def __init__(
            self,
            params,
            vocabulary,
            utter_attention_key_size,
            schema_attention_key_size):
        TokenPredictor.__init__(self, params, vocabulary, utter_attention_key_size)

        if params.use_schema_attention:
            self.utterance_attention_module = self.attention_module
            self.schema_attention_module = Attention(params.decoder_state_size, schema_attention_key_size, schema_attention_key_size)

        if self.params.use_query_attention:
            self.query_attention_module = Attention(params.decoder_state_size, params.encoder_state_size, params.encoder_state_size)
            self.start_query_attention_vector = torch_utils.add_params((params.encoder_state_size,), "start_query_attention_vector")

        if params.use_schema_attention and self.params.use_query_attention:
            self.state_transform_weights = torch_utils.add_params((params.decoder_state_size + utter_attention_key_size + schema_attention_key_size + params.encoder_state_size, params.decoder_state_size), "weights-state-transform")
        elif params.use_schema_attention:
            self.state_transform_weights = torch_utils.add_params((params.decoder_state_size + utter_attention_key_size + schema_attention_key_size, params.decoder_state_size), "weights-state-transform")

        # Use lstm schema encoder
        self.schema_token_weights = torch_utils.add_params((params.decoder_state_size, schema_attention_key_size), "weights-schema-token")

        if self.params.use_previous_query:
            self.query_token_weights = torch_utils.add_params((params.decoder_state_size, self.params.encoder_state_size), "weights-query-token")

        if self.params.use_copy_switch:
            if self.params.use_query_attention:
                self.state2copyswitch_transform_weights = torch_utils.add_params((params.decoder_state_size + utter_attention_key_size + schema_attention_key_size + params.encoder_state_size, 1), "weights-state-transform")
            else:
                self.state2copyswitch_transform_weights = torch_utils.add_params((params.decoder_state_size + utter_attention_key_size + schema_attention_key_size, 1), "weights-state-transform")

    def _get_schema_token_scorer(self, state):
        return torch.t(torch_utils.linear_layer(state, self.schema_token_weights))

    def _get_query_token_scorer(self, state):
        return torch.t(torch_utils.linear_layer(state, self.query_token_weights))

    def _get_copy_switch(self, state):
        return torch.sigmoid(torch_utils.linear_layer(state, self.state2copyswitch_transform_weights)).squeeze()

    def forward(self, prediction_input, dropout=0.):
        decoder_state = prediction_input.decoder_state
        input_hidden_states = prediction_input.input_hidden_states

        input_schema = prediction_input.input_schema
        schema_states = prediction_input.schema_states

        if self.params.use_schema_attention:
            schema_attention_results = self.schema_attention_module(decoder_state, schema_states)
            utterance_attention_results = self.utterance_attention_module(decoder_state, input_hidden_states)
        else:
            utterance_attention_results = self.attention_module(decoder_state, input_hidden_states)
            schema_attention_results = None

        query_attention_results = None
        if self.params.use_query_attention:
            previous_query_states = prediction_input.previous_query_states
            if len(previous_query_states) > 0:
                query_attention_results = self.query_attention_module(decoder_state, previous_query_states[-1])
            else:
                query_attention_results = self.start_query_attention_vector
                query_attention_results = AttentionResult(None, None, query_attention_results)

        if self.params.use_schema_attention and self.params.use_query_attention:
            state_and_attn = torch.cat([decoder_state, utterance_attention_results.vector, schema_attention_results.vector, query_attention_results.vector], dim=0)
        elif self.params.use_schema_attention:
            state_and_attn = torch.cat([decoder_state, utterance_attention_results.vector, schema_attention_results.vector], dim=0)
        else:
            state_and_attn = torch.cat([decoder_state, utterance_attention_results.vector], dim=0)

        intermediate_state = self._get_intermediate_state(state_and_attn, dropout=dropout)
        vocab_scores, vocab_tokens = self._score_vocabulary_tokens(intermediate_state)

        final_scores = vocab_scores
        aligned_tokens = []
        aligned_tokens.extend(vocab_tokens)

        schema_states = torch.stack(schema_states, dim=1)
        schema_scores, schema_tokens = score_schema_tokens(input_schema, schema_states, self._get_schema_token_scorer(intermediate_state))

        final_scores = torch.cat([final_scores, schema_scores], dim=0)
        aligned_tokens.extend(schema_tokens)

        # Previous Queries
        previous_queries = prediction_input.previous_queries
        previous_query_states = prediction_input.previous_query_states

        copy_switch = None
        query_scores = None
        query_tokens = None
        if self.params.use_previous_query and len(previous_queries) > 0:
            if self.params.use_copy_switch:
                copy_switch = self._get_copy_switch(state_and_attn)
            for previous_query, previous_query_state in zip(previous_queries, previous_query_states):
                assert len(previous_query) == len(previous_query_state)
                previous_query_state = torch.stack(previous_query_state, dim=1)
                query_scores, query_tokens = score_query_tokens(previous_query, previous_query_state, self._get_query_token_scorer(intermediate_state))
                query_scores = query_scores.squeeze()

        final_scores = final_scores.squeeze()

        return TokenPrediction(final_scores, aligned_tokens, utterance_attention_results, schema_attention_results, query_attention_results, copy_switch, query_scores, query_tokens, decoder_state)
