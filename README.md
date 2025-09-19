
First you need to calculate scores for the words in each document.

How BM-25 Works:
```
Let's say you calculated the BM25 score for each of the words in each document

Now you want to find the most relevant document related to a word

Document with highest BM25 weight for a word across all documents should be returned

If we have multiple words as a query, we can have the summation of BM25 scores across the document in which they coexist and return the highest scoring document first
```


### References

- https://www.kaggle.com/code/ashishkumarak/bm-25-from-scratch
- https://www.youtube.com/watch?v=D3yL63aYNMQ&ab_channel=StanfordOnline
