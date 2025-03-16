import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAction, QApplication, QLabel, QMainWindow, QToolBar, QTextEdit, QGridLayout, QWidget,QFileDialog, QPlainTextEdit
from PyQt5.QtGui import QIcon


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Test_1")

        '''<--------------------------------------------  Cutting line  -------------------------------------------->'''
        '''Label area'''
        # Generate labels

        listOftext = QLabel('List of text', self)
        listOftext.move(10, 20)
        file = QLabel('Files:', self)
        file.move(10, 35)
        totalWords = QLabel('Total words:', self)
        totalWords.move(10, 50)

        '''<--------------------------------------------  Cutting line  -------------------------------------------->'''
        '''Menu bar area'''
        # Generate the Menu bar

        menubar = self.menuBar()

        # Sub class
        fileMenu = menubar.addMenu('File')
        modelsMenu = menubar.addMenu('Models')
        analysis = menubar.addMenu('Analysis')
        parameter = menubar.addMenu('Paramter setting')

        '''<--------------------------------------------  Cutting line  -------------------------------------------->'''

        # 1. File
        open_file = QAction('Open File', self)
        exit = QAction('Exit', self)

        fileMenu.addAction(open_file)
        fileMenu.addSeparator()
        fileMenu.addAction(exit)

        open_file.triggered.connect(self.open_file_function)
        exit.triggered.connect(self.exit_function)

        '''Tool bar area'''
        # toolbar = QToolBar("My main toolbar")
        # self.addToolBar(toolbar)
        # # 1. File button
        # button_action = QAction("File", self)
        # button_action.setStatusTip("This is your button")
        # button_action.triggered.connect(self.drop_down_list)
        # toolbar.addAction(button_action)

        '''Layout'''
        self.main_widget = QWidget()  # Create main part
        self.main_layout = QGridLayout()
        self.main_widget.setLayout(self.main_layout)

        self.left_widget = QWidget()  # Create left widget
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QGridLayout()
        self.left_widget.setLayout(self.left_layout)

        self.right_widget = QWidget()  # Create right widget
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QGridLayout()
        self.right_widget.setLayout(self.right_layout)

        self.main_layout.addWidget(self.left_widget, 0, 0, 12, 2)
        self.main_layout.addWidget(self.right_widget, 0, 2, 12, 10)
        self.setCentralWidget(self.main_widget)

        self.textEdit = QPlainTextEdit()
        self.right_layout.addWidget(self.textEdit, 0, 0, 4, 8)

        self.setGeometry(300, 300, 1200, 700)
        self.setWindowIcon(QIcon('icon.png'))
        self.setWindowTitle('Text Analyzer')

    def open_file_function(self):
        # fname = QFileDialog.getOpenFileName(self, 'Open file', '.')
        filename = QFileDialog.getOpenFileName(self, "Open file", ".")
        path = filename[0]
        with open(path,"r") as f:
            data = f.read()
            self.textEdit.setPlainText(data)


    def exit_function(self):
        print("Exit")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
