# Meta-Aug

### A document-level data augmentation framework (Meta-Aug) leverages various meta-data to enrich the document representation for downstream tasks. 

## A quick guide of using _Meta-Aug_

1. Calculate the scores of used reviews' attributes.
   Example:
```
  from core.prep_meta import cal_len_atr
  from core.prep_meta import cal_pos_atr
  
  # calculate length attribute
  len_atr = np.array(list(map(cal_len_atr, review_text)))
  len_atr = len_atr.reshape(-1, 1)
            
  # calculate part_of_speech attribute
  pos_atr = np.array(list(map(cal_pos_atr, sentences)))
  atr_scores = np.concatenate((len_atr,pos_atr), axis = 1)
```
2. Update the data_loader and return the attribute scores alongside with the text data
   Example:
```
  if self.use_meta:
            return (
                torch.Tensor(review_text_data),
                torch.Tensor(atr_scores).type(torch.float),
                ...
            )
        else:
            return(
                torch.Tensor(review_text_data),
                None,
                ...
            )
```
3. Implement the Meta-Trans transformer to encode the review attribute information.
```
  output, attn_output_weights = self.meta_trans(batch_text_data, batch_meta, ...)
```
