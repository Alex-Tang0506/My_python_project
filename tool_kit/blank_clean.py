import os
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('wordnet')

path = os.getcwd()

book_list = ["Daniel Deronda(1876)"]
print(path)

for i in range(0, len(book_list)):
    with open(path + r'\{}.txt'.format(book_list[i]), 'r') as f:
        data=f.readlines()
        for line in data:
            odom=line.split()
            tmp_str = "".join(odom)
            result =  ' '.join(tmp_str.split())

with open((os.path.join('test_copy.txt')), 'w') as f:
    f.write(result)