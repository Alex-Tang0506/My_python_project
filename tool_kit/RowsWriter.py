import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

path = os.getcwd()

book_list = ["Nicola's-cleaned-up-novel"]
print(path)

def TXTRead_Writeline(BookList):
    for i in range(0, len(BookList)):
        with open(path + r'\{}.txt'.format(BookList[i]), 'r') as f:
            lines = f.readlines()
            print(lines)
            with open(path + r'\{}_Rows.txt'.format(BookList[i]), 'a+') as f_1:
                _n = 0
                for content in lines:
                    f_1.write(str(_n + 1) + ':' + " " + content)
                    print(content)
                    _n = _n + 1
                    # content = str(i) + "." + "  " + content
    print("Finished")

