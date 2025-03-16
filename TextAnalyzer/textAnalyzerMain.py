'''
This function is the basic skeleton of text-analyzer UI/interface,
The purpose is to realize some basic operation functions.

Pyqt 5 tutorial/test
Author : Xianming Tang
Date: 03/21/2022
'''
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        menubar = self.menuBar()

        # create some new menus on the menu bar
        fileMenu = menubar.addMenu('File')
        modelsMenu = menubar.addMenu('Models')
        analysis = menubar.addMenu('Analysis')
        parameter = menubar.addMenu('Paramter setting')

        # 1. file menu
        # add 'import' sub menu, this menu still have a sub menu 'import txt'
        sub_fileMenu = QAction('Import...',self)
        fileMenu.addAction(sub_fileMenu)

        # space
        fileMenu.addSeparator()
        fileMenu.addAction(QAction('Exit', self))

        # 2. models menu
        sub_models_TFIDF = QAction('TF-RDF model', self)
        sub_models_EM = QAction('EM model', self)
        modelsMenu.addAction(sub_models_TFIDF)
        modelsMenu.addAction(sub_models_EM)

        # 3. analysis menu
        analysis.addAction(QAction('Analysis', self))

        # 4. Parameter setting
        sub_para_TDIDF = QAction('TF-IDF',self)
        sub_para_EM = QAction('EM', self)
        parameter.addAction(sub_para_TDIDF)
        parameter.addAction(sub_para_EM)

        # 4. Some information
        listOftext = QLabel('List of text',self)
        listOftext.move(10,20)
        file = QLabel('Files:',self)
        file.move(10,35)
        totalWords = QLabel('Total words:',self)
        totalWords.move(10,50)

        self.setGeometry(300, 300, 800, 500)
        self.setWindowIcon(QIcon('icon.png'))
        self.setWindowTitle('Text Analyzer')

        # list of text area
        self.textbox = QLineEdit(self)
        self.textbox.move(10, 80)
        self.textbox.resize(100, 400)
        self.show()


def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
