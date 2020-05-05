"""Modified from torchnlp.nn.attention module."""
import torch
import torch.nn as nn


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.
    
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`
    """

    def __init__(self, query_dim, context_dim, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(query_dim, context_dim, bias=False)

        # self.linear_out = nn.Linear(dimensions*2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        # self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [output length, batch size, query_dim]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [query length, batch size, context_dim]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [output length, batch size, context_dim]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [output length, batch size, query length]):
              Tensor containing attention weights.
        """
        query = query.transpose(0, 1)
        context = context.transpose(0, 1)
        batch_size, output_len, query_dim = query.size()
        _, query_len, context_dim = context.size()

        if self.attention_type == "general":
            query = query.reshape(batch_size*output_len, query_dim)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, context_dim)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, context_dim) * (batch_size, query_len, context_dim) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size*output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, context_dim) ->
        # (batch_size, output_len, context_dim)
        output = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        # combined = torch.cat((mix, query), dim=2)
        # combined = combined.view(batch_size*output_len, 2*dimensions)

        # # Apply linear_out on every 2nd dimension of concat
        # # output -> (batch_size, output_len, dimensions)
        # output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        # output = self.tanh(output)

        return output.transpose(0, 1)

if __name__ == "__main__":
    att = Attention(10, 20)
    q = torch.randn(5, 3, 10)
    c = torch.randn(7, 3, 20)
    print(att(q, c).size())