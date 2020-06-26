"""Modified from torchnlp.nn.attention module."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.
    
    Args:
        dimensions (int): Dimensionality of the query and context.
        attn_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`
    """

    def __init__(self, query_dim, context_dim, attn_type='general'):
        super(Attention, self).__init__()

        if attn_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attn_type = attn_type
        if self.attn_type == 'general':
            self.linear_in = nn.Linear(query_dim, context_dim, bias=False)

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [output length, batch size, query_dim]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [context length, batch size, context_dim]): Data
                overwhich to apply the attention mechanism.

        Returns:
            output (:class:`torch.LongTensor` [output length, batch size, context_dim]):
                Tensor containing the attended features.
        """
        query = query.transpose(0, 1)
        context = context.transpose(0, 1)
        batch_size, output_len, query_dim = query.size()
        _, context_len, context_dim = context.size()

        if self.attn_type == "general":
            query = query.reshape(batch_size*output_len, query_dim)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, context_dim)

        # (batch_size, output_len, context_dim) * (batch_size, context_dim, context_len) ->
        # (batch_size, output_len, context_len)
        attn_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attn_scores = attn_scores.view(batch_size*output_len, context_len)
        attn_weights = F.softmax(attn_scores, -1)
        attn_weights = attn_weights.view(batch_size, output_len, context_len)

        # (batch_size, output_len, context_len) * (batch_size, context_len, context_dim) ->
        # (batch_size, output_len, context_dim)
        output = torch.bmm(attn_weights, context)

        return output.transpose(0, 1)

if __name__ == "__main__":
    att = Attention(10, 20)
    q = torch.randn(5, 3, 10)
    c = torch.randn(7, 3, 20)
    print(att(q, c).size())