import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from core.meta_trans import MetaTrans
from core.config import prep_config

class BertClassifier(nn.Module):
    def __init__(self, requires_grad = True):
        super(BertClassifier, self).__init__()
        
        # load model configuration
        self.config_path = "../../configs/IMDB_senti_cls.json"
        self.config = prep_config(self.config_path)
        self.num_labels = self.config['model']['num_of_labels']
        self.num_atr = len(self.config['model']['meta_atr'])
        self.max_token_len = self.config['model']['max_token_len']
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        self.feat_dim = self.config['model']['emb_dim']
        self.lin_proj = nn.Linear(self.feat_dim * self.max_token_len, self.num_labels)
        self.loss_fct = CrossEntropyLoss()
        self.meta_trans = MetaTrans(self.feat_dim, self.num_atr)
        for param in self.bert.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

    def forward(self, batch_feats, batch_mask, labels, batch_meta=None):
        output, attn_output_weights = self.meta_trans(batch_feats, batch_mask, batch_meta)
        # output dim: batch_size x num_tokens x feat_dim
        batch_size = output.size(0)
        output = output.reshape(batch_size, -1)
        logits = self.linear_layer(output)
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities