import nltk

def cal_len_atr(text, max_len = 500):
    return len(text.split()) / (1.0 * max_len)

def cal_pos_atr(text):
    # calculate the ratio of noun, verb, adverb and adjective
    tokens = nltk.word_tokenize(text)
    num_of_tokens = len(tokens)
    if num_of_tokens == 0:
        return [0.0, 0.0, 0.0, 0.0]
    tagged_tokens = nltk.pos_tag(tokens)
    num_of_noun = 0
    num_of_verb = 0
    num_of_adv = 0
    num_of_adj = 0
    for (word, tag) in tagged_tokens:
        if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
            num_of_noun += 1
        elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            num_of_verb += 1
        elif tag in ['RB', 'RBR', 'RBS']:
            num_of_adv += 1
        elif tag in ['JJ', 'JJR', 'JJS']:
            num_of_adj += 1
    return [num_of_noun / (1.0 * len(tokens)), num_of_verb / (1.0 * len(tokens)),
            num_of_adv / (1.0 * len(tokens)), num_of_adj / (1.0 * len(tokens))]