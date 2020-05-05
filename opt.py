    def build_optim(self):
        params_trainer = []
        params_bert_trainer = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'model_bert' in name:
                    params_bert_trainer.append(param)
                else:
                    params_trainer.append(param)
        self.trainer = torch.optim.Adam(params_trainer, lr=self.params.initial_learning_rate)
        if self.params.fine_tune_bert:
            self.bert_trainer = torch.optim.Adam(params_bert_trainer, lr=self.params.lr_bert)

    def loss_turn(
            self,
            gold_query=None,
            ):
        """Calculates loss for a single turn."""
        token_accuracy = 0.
        decoder_results = None

        all_scores = []
        all_alignments = []
        for prediction in decoder_results.predictions:
            scores = F.softmax(prediction.scores, dim=0)
            alignments = prediction.aligned_tokens
            if self.params.use_prev_query and self.params.use_copy_switch and len(prev_output_seqs) > 0:
                query_scores = F.softmax(prediction.query_scores, dim=0)
                copy_switch = prediction.copy_switch
                scores = torch.cat([scores * (1 - copy_switch), query_scores * copy_switch], dim=0)
                alignments = alignments + prediction.query_tokens

            all_scores.append(scores)
            all_alignments.append(alignments)

        # Compute the loss
        gold_seq = gold_query

        loss = torch_utils.compute_loss(gold_seq, all_scores, all_alignments, get_token_indices)
        if not training:
            predicted_seq = torch_utils.get_seq_from_scores(all_scores, all_alignments)
            token_accuracy = torch_utils.per_token_accuracy(gold_seq, predicted_seq)
        fed_seq = gold_seq

        return loss, token_accuracy, decoder_states