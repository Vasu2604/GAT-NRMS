from torch.cuda.amp import autocast
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class DotProductClickPredictor(torch.nn.Module):
    def __init__(self):
        super(DotProductClickPredictor, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, candidate_news_vector, user_vector):
        """
        Args:
            candidate_news_vector: batch_size, candidate_size, X
            user_vector: batch_size, X
        Returns:
            (shape): batch_size
        """
        # batch_size, candidate_size
        candidate_news_vector = candidate_news_vector.to(self.device)
        user_vector = user_vector.to(self.device)

        # batch_size, candidate_size
        probability = torch.bmm(candidate_news_vector,
                                user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        return probability
                            

class AdditiveAttention(nn.Module):
    def __init__(self, query_vector_dim, candidate_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.query_vector = nn.Parameter(torch.empty(query_vector_dim, 1, device=self.device))
        nn.init.xavier_uniform_(self.query_vector)
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim).to(self.device)

    def forward(self, candidate_vector):
        temp = torch.tanh(self.linear(candidate_vector))
        candidate_weights = F.softmax(torch.matmul(temp, self.query_vector), dim=1)
        return torch.sum(candidate_vector * candidate_weights, dim=1)


class GATNRMS(nn.Module):
    def __init__(self, config, pretrained_word_embedding=None):
        super(GATNRMS, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding).to(self.device)
        self.user_encoder = UserEncoder(config).to(self.device)
        self.click_predictor = DotProductClickPredictor().to(self.device)

    def forward(self, candidate_news, clicked_news):
        with autocast():
            candidate_news = [{k: v.to(self.device) for k, v in news.items()} for news in candidate_news]
            clicked_news = [{k: v.to(self.device) for k, v in news.items()} for news in clicked_news]

            candidate_news_vector = torch.stack([self.news_encoder(x) for x in candidate_news], dim=1)
            clicked_news_vector = torch.stack([self.news_encoder(x) for x in clicked_news], dim=1)
            user_vector = self.user_encoder(clicked_news_vector)
            return self.click_predictor(candidate_news_vector, user_vector)

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                },
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.user_encoder(clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size, word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        # candidate_size
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)

class NewsEncoder(nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words, config.word_embedding_dim, padding_idx=0).to(self.device)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding, freeze=False, padding_idx=0).to(self.device)

        self.gat = GraphAttentionLayer(config.word_embedding_dim, config.word_embedding_dim, alpha=0.2).to(self.device)
        self.attention = AdditiveAttention(config.query_vector_dim, config.word_embedding_dim).to(self.device)

    def forward(self, news):
        news_vector = F.dropout(self.word_embedding(news["title"].to(self.device)),
                                p=self.config.dropout_probability,
                                training=self.training)

        batch_size, seq_len, embed_dim = news_vector.size()
        news_vector_flat = news_vector.view(-1, embed_dim)

        adj = torch.ones(batch_size * seq_len, batch_size * seq_len, device=self.device)
        news_vector_gat = self.gat(news_vector_flat, adj).view(batch_size, seq_len, -1)

        return self.attention(news_vector_gat)

class UserEncoder(nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gat = GraphAttentionLayer(config.word_embedding_dim, config.word_embedding_dim, alpha=0.2).to(self.device)
        self.attention = AdditiveAttention(config.query_vector_dim, config.word_embedding_dim).to(self.device)

    def forward(self, user_vector):
        batch_size, num_clicked_news, embed_dim = user_vector.size()
        user_vector_flat = user_vector.reshape(-1, embed_dim)

        try:
            adj = torch.ones(batch_size * num_clicked_news, batch_size * num_clicked_news, device=self.device)
        except RuntimeError:
            print(f"batch_size: {batch_size}, num_clicked_news: {num_clicked_news}")
            raise RuntimeError
        user_vector_gat = self.gat(user_vector_flat, adj).reshape(batch_size, num_clicked_news, -1)

        return self.attention(user_vector_gat)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features), device=self.device))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1), device=self.device))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        # Use sparse operations if possible
        if adj.is_sparse:
            attention = torch.sparse.softmax(adj * e, dim=1)
        else:
            zero_vec = -9e15*torch.ones_like(e, device=self.device)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)

        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


