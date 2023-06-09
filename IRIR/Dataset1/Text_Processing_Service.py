# import required packages
from flask import Flask, request
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict




app = Flask(__name__)

# Step 1: Case normalization 
def normalize_case(text):
    return text.lower().replace('.', '').replace("'", '')
   

# Step 2: Tokenization
def Tokenize_case(text):
    return word_tokenize(text)
# الخرج مصفوفة تكونات


# Step 3: Stopword removal
def removeStopWords_case(words): 
    stop_words = set(stopwords.words('english'))
    stopWords_filtered_words = []
    for w in words:
    
        if not w in stop_words and len(w) > 2: 
            stopWords_filtered_words.append(w)
        #    The isalnum() method returns True if all the characters are alphanumeric, meaning alphabet letter (a-z) and numbers (0-9).
         #   Example of characters that are not alphanumeric (Punctuation Marks): (space)!#%&? etc.
    Punctuation_Marks_filtered_words = [word for word in stopWords_filtered_words if word.isalnum()]
    return Punctuation_Marks_filtered_words
        # الخرج مصفوفة تكونات


# Step 4: Stemming
def stem_tokens_case(words):
    stemmed_words = []
    ps = PorterStemmer()
    for w in words:
        stemmed_word = ps.stem(w)
        stemmed_words.append(stemmed_word)
    return stemmed_words
# لخرج مصفوفة تكونات


# Step 5: Lemmatization
def lemmatize_tokens_case(words,pos_tags): 
    lemmatized_words = []
    lemmatizer = WordNetLemmatizer()
  
    for w in words:
        lemmatized_word = lemmatizer.lemmatize(w)
        lemmatized_words.append(lemmatized_word)
    return lemmatized_words


def process_documents(input_Path, output_Path, linesNumberYouWant):
    count = 0 
    writeFile = open(output_Path,'w')
    with open(input_Path, "r") as file: 
        # line by line 
        for i, line in enumerate(file): 
            if i > linesNumberYouWant: 
                break
            # Remove spaces at the beginning and at the end of the string:
            line = line.strip()
            # print(line)
            normalized_text = normalize_case(line)
            # print(normalized_text)
            tokens = Tokenize_case(normalized_text)
            # print(tokens)
            pos_tags = nltk.pos_tag(tokens)
            # print(pos_tags)
            filtered_tokens  = removeStopWords_case(tokens)
            # print(filtered_tokens)
            stemmed_tokens = stem_tokens_case(filtered_tokens)
            # print(stemmed_tokens)
            lemmatized_tokens = lemmatize_tokens_case(stemmed_tokens,pos_tags)
            # print(lemmatized_tokens)
            if len(lemmatized_tokens) < 2:
                continue
          
            # count how many lines stored in output file
            count += 1
            # write doc_id to output file
            writeFile.write(tokens[0])
            writeFile.write(' ')
            # remove doc_id from tokens
            lemmatized_tokens.pop(0)
            # write document content to output file as tokens
            #   الغي المصفوفة صار عندي text string
            text = str.join(' ', lemmatized_tokens)
            writeFile.write(text.strip() + '\n')

        writeFile.close()


def processQuery(query):
   
    query = query.strip()
    normalized_text = normalize_case(query)
    tokens = Tokenize_case(normalized_text)
    filtered_tokens  = removeStopWords_case(tokens)
    stemmed_tokens = stem_tokens_case(filtered_tokens)
    lemmatized_tokens = lemmatize_tokens_case(stemmed_tokens,[])
    return lemmatized_tokens


def fileToDict (filePath, linesNumber):
     #dic = defaultdict()
     dic = {}
     with open(filePath, "r") as file: 
        for i, line in enumerate(file):
            #line = next(file)
            if i > linesNumber:
                break      
            line = line.strip()
            line = line.split(' ')
            doc_id = line[0]
            line.pop(0)
            line = str.join(' ', line)
            dic[doc_id] = line
     return dic





@app.route("/", methods=['POST','GET'])
def processData():
    if (request.method == 'POST'):
        inputFilename = request.form['inputFilename']
        outputFilename = request.form['outputFilename']
        docsNumber = request.form['docsNumber']
        return process_documents(inputFilename, outputFilename, docsNumber)

if (__name__) == "__main__":
    app.run()