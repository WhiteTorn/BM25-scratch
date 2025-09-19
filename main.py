import re
import numpy as np
import math
import pandas as pd

corpus = [
    'this is the first document.',
    'this is the first ',
   'this document is the second document.',
    'and this is the third one.',
    'is this the first document?',
    'this is not even fourth document',
 ]
 
def remove_punctuation(text):
  """Removes all punctuation from a string.

  Args:
    text: The string to remove punctuation from.

  Returns:
    A string with all punctuation removed.
  """

  pattern = re.compile(r'[^\w\s]')
  return pattern.sub(' ', text)

# print(remove_punctuation(corpus[0]))

# Get unique words

unique_words = set()
for i in corpus: # go through all sentence
  only_txt = remove_punctuation(i)
  words = only_txt.split(" ")
  for j in range(len(words)-1):
    words[j] = words[j].lower()
    unique_words.add(words[j])
    
unique_words_list = list(unique_words)
# print(unique_words_list)

def tf(corpus):
  tf_dict = {}
  for i, doc in enumerate(corpus):
    txt = remove_punctuation(doc).split(" ")[:-1]
    len_doc = len(txt)
    for j, word in enumerate(unique_words_list):
      tf_dict[(i,j)] = txt.count(word)/len_doc
      
  return tf_dict
  
tf_dict = tf(corpus)

# print(tf_dict)

# Printing TF Values

def print_words_tf():
  for l,k in tf_dict.keys():
    print(unique_words_list[k], tf_dict[(l, k)])
    
# print_words_tf()

def avgdoclen(corpus):
  count = 0
  for i in corpus:
    i = remove_punctuation(i)
    count += len(i.split(' '))
  return count/len(corpus)

# print(avgdoclen(corpus))

# Scoring

def scoring(k = 1.2, b = 0.75, unique_words_list=unique_words_list, corpus = corpus):
  scoring_dict = {}
  for i, doc in enumerate(corpus):
    txt = remove_punctuation(doc).split(' ')[:-1]
    len_doc = len(txt)
    for j, word in enumerate(unique_words_list):
      tfval = txt.count(word)/len_doc
      scoring_dict[(i, j)] = tfval * (k+1)/(tfval + k*(1-b+b*len_doc/avgdoclen(corpus)))
  return scoring_dict
  
scoring_dict = scoring()


def IDF_BM25(word, s = 0.5):
  count = 0
  for doc in corpus:
    txt = remove_punctuation(doc).split(' ')[:-1]
    if word in txt:
      count += 1
  
  return np.log(1 + (len(corpus) - count + s)/(count+s))
  
IDF_BM25_dict = {}

for k, word in enumerate(unique_words_list):
  IDF_BM25_dict[k] = IDF_BM25(word)
  
# IDF_BM25_dict

def print_words_idf():
    for i in range(len(unique_words_list)) : 
        print(unique_words_list[i], IDF_BM25_dict[i])
# print_words_idf()

bm25_weight_dict = {}
for wd_tuple in tf_dict.keys():
  bm25_weight = scoring_dict[wd_tuple]*IDF_BM25_dict[wd_tuple[1]]
  bm25_weight_dict[wd_tuple] = bm25_weight
  
class BM25() : 
    def __init__(self, s= 0.5, k = 1.2, b = 0.4) : 
        self.feature_names = None 
        self.k =k
        self.b = b
        self.s = s
        self.avglen = None
    
    def avgdoclen(self, corpus) :  
        count = 0
        for i in corpus : 
            i = remove_punctuation(i)
            count += len(i.split(' '))
        self.avglen = count/len(corpus)
        return self.avglen
        
        
    def fit_transform(self, corpus) : 
        unique_words = set()
        for i in corpus : 
            only_txt = remove_punctuation(i)
            words = only_txt.split(' ')
            for j in range(len(words)-1) : 
                words[j] = words[j].lower()
                unique_words.add(words[j])   
        self.feature_names = list(unique_words)
        # Scoring_dict
        scoring_dict = {}
        for i,doc in enumerate(corpus) : 
            txt = remove_punctuation(doc).split(' ')[:-1]
            len_doc = len(txt)
            for j, word in enumerate(unique_words_list) : 
                    tfval = txt.count(word)/len_doc
                    scoring_dict[(i,j)] = tfval*(self.k+1)/(tfval + (self.k)*(1-(self.b)+(self.b)*len_doc/(avgdoclen(corpus))))
        # IDF_BM25
        IDF_BM25_dict = {}
        for k, word in enumerate(self.feature_names): 
            count = 0 
            for doc in corpus :
                txt = remove_punctuation(doc).split(' ')[:-1]
                if word in txt :
                    count+=1    
            IDF_BM25_dict[k]= np.log(1+ (len(corpus) - count + self.s)/(count + self.s))
        # BM_25 weight
        bm25_weight_dict = {}
        for wd_tuple in tf_dict.keys():
            bm25_weight = scoring_dict[wd_tuple]*IDF_BM25_dict[wd_tuple[1]]
            bm25_weight_dict[wd_tuple] = bm25_weight
        return bm25_weight_dict
        
             
    def print_feature_names(self):
        return self.feature_names
        
        

bm25_instance = BM25() 
X = bm25_instance.fit_transform(corpus)
bm25_instance.print_feature_names()

# print(X)

def transformation(corpus, IDF_BM25_dict, bm25_weight_dict):
  my_df = pd.DataFrame()
  term_doc_array = np.zeros((len(corpus), len(IDF_BM25_dict.keys())))
  for i in range(len(corpus)):
    for j in range(len(IDF_BM25_dict.keys())):
      term_doc_array[i][j] = bm25_weight_dict[(i, j)]
  return term_doc_array
  
term_doc_array = transformation(corpus, IDF_BM25_dict, bm25_weight_dict)

my_df = pd.DataFrame(term_doc_array, columns = unique_words_list)




def search_doc(query, top_n_docs = 5):
  query = remove_punctuation(query)
  #     words = query.split(' ') #Splitting the query if it contains more than one word 
#     word_doc_list = []
#     for word in words : ## Searching over the words of the query
  if query in unique_words_list:
    word_index =unique_words_list.index(query)
    for i, j in X:
      filtered_dict = {key: value for key, value in X.items() if key[1] == word_index}
      sorted_dict = dict(sorted(filtered_dict.items(), key=lambda x: x[1], reverse=True))
    docs = [corpus[doc_id[0]] for doc_id in list(sorted_dict.keys())[0 : top_n_docs]] ## Getting all the relevant docs in sorted manner
    #         word_doc_list.append(docs)
    #     if len(words) == 1 : 
    #         return(word_doc_list[0])
    #     else : 
    #         docs = set(word_doc_list[0]).intersection(*word_doc_list[1:]) ## If the words are in all docs
    #         if docs is not None : 
    #             return docs
    #         else : 
    #             return (word_doc_list[0])
    return docs
  else :
    print('No such word in any document')


print(search_doc('document'))

