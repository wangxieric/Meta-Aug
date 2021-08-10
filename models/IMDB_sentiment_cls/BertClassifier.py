import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from core.meta_trans import MetaTrans
from core.config import prep_config
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, requires_grad = True):
        super(BertClassifier, self).__init__()
        
        # load model configuration
        self.config_path = "../../configs/IMDB_senti_cls.json"
        self.config = prep_config(self.config_path)
        self.num_labels = self.config['model']['num_of_labels']
        self.num_atr = len(self.config['model']['meta_atr'])
        self.max_token_len = self.config['model']['max_seq_len']
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        self.feat_dim = self.config['model']['emb_dim']
        self.relu = nn.ReLU(inplace=True)
        self.lin_proj = nn.Linear(self.feat_dim * self.max_token_len, self.feat_dim)
        self.lin_pred = nn.Linear(self.feat_dim, self.num_labels)
        self.loss_fct = CrossEntropyLoss()
        
        # initialise transformer
        self.bert = BertModel.from_pretrained('bert-large-uncased')
        for param in self.bert.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient
        
        self.meta_trans = MetaTrans(self.feat_dim, self.num_atr)
        for param in self.meta_trans.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

    def forward(self, batch_seqs_ids, batch_mask, labels, batch_meta=None):
        outputs = self.bert(input_ids=batch_seqs_ids, attention_mask=batch_mask)
        
        # obtain hidden state of each token / '[cls]' token 
        # batch_feats = outputs.last_hidden_state
        
        batch_feats = outputs.last_hidden_state[:,0]
        batch_feats = batch_feats.unsqueeze(1)
        
        # set batch_mask to None for using '[cls]'-based global embedding
        batch_mask = None
        
        output, attn_output_weights = self.meta_trans(batch_feats, batch_mask, batch_meta)
        
        # output dim: batch_size x num_tokens x feat_dim
        batch_size = output.size(0)
        output = output.reshape(batch_size, -1)
        
        output = self.relu(output)
        logits = self.lin_pred(output)
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities