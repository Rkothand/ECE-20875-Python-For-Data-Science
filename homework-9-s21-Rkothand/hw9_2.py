from helper import remove_punc
import nltk
import numpy as np
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
import math 

#Clean and lemmatize the contents of a document
#Takes in a file name to read in and clean
#Return a list of words, without stopwords and punctuation, and with all words stemmed
# NOTE: Do not append any directory names to doc -- assume we will give you
# a string representing a file name that will open correctly
def readAndCleanDoc(doc) :
    nltk.download('punkt')
    #1. Open document, read text into *single* string
    txt = open(doc, 'r')
    line = txt.read()
    #2. Tokenize string using nltk.tokenize.word_tokenize
    token=nltk.tokenize.word_tokenize(line)
    #3. Filter out punctuation from list of words (use remove_punc)
    removedPunc=remove_punc(token)
    #4. Make the words lower case
    lowerCase=[i.lower() for i in removedPunc]

    #5. Filter out stopwords
    nltk.download('stopwords')
    stop = stopwords.words('english')
    docs_token_filter =[i for i in lowerCase if i not in stop]

    #6. Stem words
    nltk.download('wordnet')
    stemmer=PorterStemmer()
    words = [stemmer.stem(i) for i in docs_token_filter]
    return words
    
#Builds a doc-word matrix for a set of documents
#Takes in a *list of filenames*
#
#Returns 1) a doc-word matrix for the cleaned documents
#This should be a 2-dimensional numpy array, with one row per document and one 
#column per word (there should be as many columns as unique words that appear
#across *all* documents. Also, Before constructing the doc-word matrix, 
#you should sort the wordlist output and construct the doc-word matrix based on the sorted list
#
#Also returns 2) a list of words that should correspond to the columns in
#docword
def buildDocWordMatrix(doclist) :
    #1. Create word lists for each cleaned doc (use readAndCleanDoc)
    txt = []
    list_word = []
    for i in doclist:
         txt.append(readAndCleanDoc(i))
   
    for j in txt:
        for k in j:
            if(k not in list_word):
                list_word.append(k)
    list_word.sort()         
    #2. Use these word lists to build the doc word matrix
    doc_word_simple = []
    for doc in txt:
       doc_vec = [0]*len(list_word) #Each document is represented as a vector
       for word in doc:
           ind = list_word.index(word)
           doc_vec[ind] += 1 #Increment the corresponding word index
           doc_word_simple.append(doc_vec)    

    doc_word_simple =np.asarray(doc_word_simple)
    return doc_word_simple, list_word
    
#Builds a term-frequency matrix
#Takes in a doc word matrix (as built in buildDocWordMatrix)
#Returns a term-frequency matrix, which should be a 2-dimensional numpy array
#with the same shape as docword
def buildTFMatrix(docword) :
    #fill in    
 
    
    [rows,columns] = (len(docword),len(docword[0]))
    tf = docword.copy()
    for l in range(rows):
         for word in range(columns):
             sum1 = sum(docword[l])
             tf[l,word]= np.divide((docword[l,word]),sum1)
    
          
    

    return tf
    
#Builds an inverse document frequency matrix
#Takes in a doc word matrix (as built in buildDocWordMatrix)
#Returns an inverse document frequency matrix (should be a 1xW numpy array where
#W is the number of words in the doc word matrix)
#Don't forget the log factor!
def buildIDFMatrix(docword) :
    #fill in
    list1 = []
    count_pos = 0
    for i in docword.transpose():
        for j in i:
            if(j > 0):
                count_pos +=1
        list1.append(count_pos)
        count_pos=0
    idf= []
    idf = np.log(np.divide(len(docword),list1))

    return idf
    
#Builds a tf-idf matrix given a doc word matrix
def buildTFIDFMatrix(docword) :
    #fill in
    tf = buildTFMatrix(docword)
    idf = buildIDFMatrix(docword)
    tfidf = np.multiply(tf,idf)

    return tfidf
    
#Find the three most distinctive words, according to TFIDF, in each document
#Input: a docword matrix, a wordlist (corresponding to columns) and a doclist 
# (corresponding to rows)
#Output: a dictionary, mapping each document name from doclist to an (ordered
# list of the three most common words in each document
def findDistinctiveWords(docword, wordlist, doclist) :
    distinctiveWords = {}
    #fill in
    #you might find numpy.argsort helpful for solving this problem:
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    

    tfidf = buildTFIDFMatrix(docword)
    for i in tfidf:
         sortedlist=argsort(i)
         indexes=sortedlist[-3:]
         top3 = []
         top3.append(wordlist[indexes[0]])
         top3.append(wordlist[indexes[1]])
         top3.append(wordlist[indexes[2]])
         distinctiveWords[i].append(top3)    
    
    return distinctiveWords


if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join, splitext
    
    ### Test Cases ###
    directory='lecs'
    path1 = join(directory, '1_vidText.txt')
    path2 = join(directory, '2_vidText.txt')
    
    # Uncomment and recomment ths part where you see fit for testing purposes
    
    print("*** Testing readAndCleanDoc ***")
    print(readAndCleanDoc(path1)[0:5])
    print("*** Testing buildDocWordMatrix ***") 
    doclist =[path1, path2]
    docword, wordlist = buildDocWordMatrix(doclist)
    print(docword.shape)
    print(len(wordlist))
    print(docword[0][0:10])
    print(wordlist[0:10])
    print(docword[1][0:10])
    print("*** Testing buildTFMatrix ***") 
    tf = buildTFMatrix(docword)
    print(tf[0][0:10])
    print(tf[1][0:10])
    print(tf.sum(axis =1))
    print("*** Testing buildIDFMatrix ***") 
    idf = buildIDFMatrix(docword)
    print(idf[0][0:10])
    print("*** Testing buildTFIDFMatrix ***") 
    tfidf = buildTFIDFMatrix(docword)
    print(tfidf.shape)
    print(tfidf[0][0:10])
    print(tfidf[1][0:10])
    print("*** Testing findDistinctiveWords ***")
    print(findDistinctiveWords(docword, wordlist, doclist))

