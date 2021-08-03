import os
import pandas as pd
import torch
import sys
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from sys import platform
import pickle
import gzip
from utils_Ext import train, validate, test, Metric
from BertClassifier import BertClassifier
from core.config import prep_config
from core.prep_meta import cal_len_atr
from core.prep_meta import cal_pos_atr



class TextDataPreprocess(Dataset):
    """
    Text Data Encoding, which generates the intial token_ids of the input text.
    """
    def __init__(self, input_data, max_seq_len = 100):
        super(TextDataPreprocess, self).__init__()
        
        # initialise transformer
        self.bert = BertModel.from_pretrained('bert-large-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        
        self.config_path = '../../configs/IMDB_senti_cls.json'
        self.config = prep_config(self.config_path)
        
        self.max_seq_len = self.config['model']['max_seq_len']
        self.use_meta = self.config['model']['use_meta']
        if self.use_meta:
            self.meta_feat = self.config['meta_feat']
            
        self.inputs = self.get_input(input_data)

    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        outputs = []
        for key, values in self.inputs.items():
            outputs.append(self.inputs[key][idx])
        return outputs
    
    def get_input(self, input_data):
        # input_data is a dataframe variable
        sentences = input_data['review'].values
        labels = input_data['label'].values
        
        tokens_seq = list(map(self.tokenizer.tokenize, sentences))
        result = list(map(self.trunate_and_pad, tokens_seq))
        
        input_ids = [i[0] for i in result]
        attention_mask = [i[1] for i in result]
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sentence_feat = outputs.last_hidden_state[:,0]
        
        if self.use_meta:
            atr_scores = []
            # calculate length attribute
            atr_scores.append(list(map(cal_len_atr, sentences)))
            
            # calculate part_of_speech attribute
            atr_scores.append(list(map(cal_pos_atr, sentences)))
            
            return (
                torch.Tensor(sentence_feat).type(torch.float),
                torch.Tensor(attention_mask).type(torch.long),
                torch.Tensor(labels).type(torch.long),
                torch.Tensor(atr_scores).type(torch.float),
            )
        else:
            return(
                torch.Tensor(sentence_feat).type(torch.float),
                torch.Tensor(attention_mask).type(torch.long),
                torch.Tensor(labels).type(torch.long),
                None,
            )
    
    def trunate_and_pad(self, tokens_seq):
        
        # add '[cls]' to the beginning
        tokens_seq = ['[cls]'] + tokens_seq
        # Length Control
        if len(tokens_seq) > self.max_seq_len:
            tokens_seq = tokens_seq[0 : self.max_seq_len]
        padding = [0] * (self.max_seq_len - len(tokens_seq))
        
        # Convert tokens_seq to token_ids
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens_seq)
        input_ids += padding
        attention_mask = [1] * len(tokens_seq) + padding
        token_types_ids = [0] * self.max_seq_len

        assert len(input_ids) == self.max_seq_len
        assert len(attention_mask) == self.max_seq_len
        assert len(token_types_ids) == self.max_seq_len
        
        return input_ids, attention_mask, token_types_ids


def model_train_validate_test(train_df, test_df, target_dir, 
         max_seq_len=100,
         epochs=20,
         batch_size=32,
         lr=2e-05,
         patience=3,
         max_grad_norm=10.0,
         if_save_model=True,
         checkpoint=None):
    """
    Parameters
    ----------
    train_df : pandas dataframe of train set.
    test_df : pandas dataframe of test set.
    target_dir : the path where you want to save model.
    max_seq_len: the max truncated length.
    epochs : the default is 3.
    batch_size : the default is 32.
    lr : learning rate, the default is 2e-05.
    patience : the default is 1.
    max_grad_norm : the default is 10.0.
    if_save_model: if save the trained model to the target dir.
    checkpoint : the default is None.
    """

    bertclassifier = BertClassifier(requires_grad = True)
    tokenizer = bertclassifier.tokenizer
    
    print(20 * "=", " Preparing for training ", 20 * "=")
    # Path to save the model, create a folder if not exist.
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    # -------------------- Data loading --------------------------------------#
    
    # For the IMDB dataset, there is no validation dataset

    print("\t* Loading training data...")
    train_data = TextDataPreprocess(tokenizer, train_df, max_seq_len = max_seq_len)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    
    print("\t* Loading test data...")
    test_data = TextDataPreprocess(tokenizer,test_df, max_seq_len = max_seq_len) 
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    
    # -------------------- Model definition ------------------- --------------#
    
    print("\t* Building model...")
    device = torch.device("cuda")
    model = bertclassifier.to(device)
    
    # -------------------- Preparation for training  -------------------------#
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {
                    'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    'weight_decay':0.01
            },
            {
                    'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    'weight_decay':0.0
            }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    ## Implement of warm up
    ## total_steps = len(train_loader) * epochs
    ## scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=60, num_training_steps=total_steps)
    
    # When the monitored value is not improving, the network performance could be improved by reducing the learning rate.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)

    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    valid_aucs = []
    
    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        train_accuracy = checkpoint["train_accuracy"]
        valid_losses = checkpoint["valid_losses"]
        valid_accuracy = checkpoint["valid_accuracy"]
        valid_auc = checkpoint["valid_auc"]
        
    # -------------------- Training epochs -----------------------------------#
    
    print("\n", 20 * "=", "Training bert model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, epoch, max_grad_norm)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)  
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%".format(epoch_time, epoch_loss, (epoch_accuracy*100)))
        
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        ## scheduler.step()
        
        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            if (if_save_model):
                  torch.save({"epoch": epoch, 
                           "model": model.state_dict(),
                           "optimizer": optimizer.state_dict(),
                           "best_score": best_score,
                           "epochs_count": epochs_count,
                           "train_losses": train_losses,
                           "train_accuracy": train_accuracies,
                           "valid_losses": valid_losses,
                           "valid_accuracy": valid_accuracies,
                           "valid_auc": valid_aucs
                           },
                           os.path.join(target_dir, "best.pth.tar"))
                  print("save model succesfully!\n")
            
            # run model on test set and save the prediction result to csv
            print("* Test for epoch {}:".format(epoch))
            _, _, test_accuracy, _, all_prob = validate(model, test_loader)
            print("Test accuracy: {:.4f}%\n".format(test_accuracy))
            test_prediction = pd.DataFrame({'prob_1':all_prob})
            test_prediction['prob_0'] = 1-test_prediction['prob_1']
            test_prediction['prediction'] = test_prediction.apply(lambda x: 0 if (x['prob_0'] > x['prob_1']) else 1, axis=1)
            test_prediction = test_prediction[['prob_0', 'prob_1', 'prediction']]
            test_prediction.to_csv(os.path.join(target_dir,"test_prediction_ext_v2.csv"), index=False)
             
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break

        
if __name__ == "__main__":
    sys.stdout = open('outputs/BERT/results_IMDB_meta.txt', 'w', buffering=1)
    data_df = pickle.load(gzip.open("data/data_processing/processed_IMDB_data.p", 'rb'))
    train_df = data_df.head(25000)
    test_df = data_df.tail(25000)
    target_dir = "outputs/BERT"
    model_train_validate_test(train_df, test_df, target_dir, max_seq_len=100, epochs=5, batch_size=32, lr=1e-5, patience=2, max_grad_norm=10.0, if_save_model=True, checkpoint=None)
    test_result = pd.read_csv(os.path.join(target_dir, 'test_prediction_ext.csv'))
    Metric(test_df.label, test_result.prediction)