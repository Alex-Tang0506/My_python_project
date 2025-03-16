import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import random

window_titles = [
    'My App',
    'Still my App',
    'What on earth',
    'Something went wrong',
    'Random title'
]


class MainWindow(QMainWindow):
    def __init__(self):
        # Inherit all of initialization settings from the QMainWindow class
        super().__init__()

        self.setWindowTitle("Text Analyser")
        self.label = QLabel()
        self.input = QLineEdit()
        self.input.textChanged.connect(self.label.setText)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.input)
        self.layout.addWidget(self.label)

        self.container = QWidget()
        self.container.setLayout(self.layout)

        # Create a button

        # Initialize the main page
        self.setCentralWidget(self.container)
        self.setFixedSize(QSize(800, 600))

    # Button clicked event
    def the_button_was_clicked(self):
        print("Clicked")
        new_window_title = random.choice(window_titles)
        print("setting title: %s" % new_window_title)
        self.setWindowTitle(new_window_title)
        if new_window_title == 'Something went wrong':
            print(new_window_title + " The button will disabled")
            self.button.setDisabled(True)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
