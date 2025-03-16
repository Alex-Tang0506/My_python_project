import os

path = os.getcwd()

book_list = ['Adam Bede(1859)_test', 'Brother Jacob(1864)_test', 'Daniel Deronda(1876)_test','Felix Holt, the Radical(1866)_test',
             'Impressions of Theophrastus Such (1879)_test','Middlemarch(1871-72)_test','Romola(1863)_test','Scenes of Clerical Life(1858)_test',
             'Silas Marner(1861)_test','The Lifted Veil(1859)_test','The Mill on the Floss(1860)_test']


def TXTRead_Writeline(BookList):
    for i in range(0, len(BookList)):
        with open(path + r'\{}.txt'.format(BookList[i]), 'r') as f:
            lines = f.readlines()
            with open(path + r'\{}_Rows.txt'.format(BookList[i]), 'a+') as f_1:
                _n = 0
                for content in lines:
                    f_1.write(str(_n + 1) + ':' + " " + content)
                    print(content)
                    _n = _n + 1
                    # content = str(i) + "." + "  " + content
    print("Finished")


TXTRead_Writeline(book_list)
