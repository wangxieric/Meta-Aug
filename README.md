# Meta-Aug

### A document-level data augmentation framework (Meta-Aug) leverages various meta-data to enrich the document representation for downstream tasks. 
#### Example Text:
```
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions 
between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. 
The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. 
The technology can then accurately extract information and insights contained in the documents as well as categorize and organize 
the documents themselves.
```
#### Example meta-data:
```
Length: number_of_tokens / max_length = 82 / 500 = 0.164
Number of Unique Terms: number_of_unique_tokens / max_length = 54 / 500 = 0.108
Noun Ratio: number_of_nouns / number_of_tokens = 14 / 82 = 0.171
Lexical diversity: VOCD => 50.48, MTLD => 40.88 
(hint: the higher the lexical diversity, the more complext the text.)
...
```

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
  from core.meta_trans import MetaTrans
  output, attn_output_weights = self.meta_trans(batch_text_data, batch_meta, ...)
```
Hint: the shape of batch_text_data is (batch_size x num_atr x num_tokens x emb_size), 
the num_tokens is 1 if the global embedding is used (e.g. the embedding of '[cls]').
