from PySide6.QtWidgets import QMainWindow, QStackedWidget, QWidget


class MainWindow(QMainWindow):
    """
    Root application window.
    Fixed to the Pi touchscreen's exact resolution: 480x320 landscape.
    """

    SCREEN_WIDTH = 320
    SCREEN_HEIGHT = 480

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Retinal Disease Detector")
        self.setFixedSize(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)  # exact match, no more/less
        self.statusBar().hide()

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #FFFFFF;
                color: #111111;
                font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', sans-serif;
            }
            QPushButton {
                background-color: #111111;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
                padding: 6px 10px;
            }
            QPushButton:hover { background-color: #333333; }
            QPushButton:pressed { background-color: #000000; }
            QPushButton.outline {
                background-color: #FFFFFF;
                color: #111111;
                border: 2px solid #111111;
            }
            QPushButton.outline:hover { background-color: #F5F5F5; }
            QLabel { color: #111111; }
        """)

    def show_screen(self, screen: QWidget) -> None:
        if self._stack.indexOf(screen) == -1:
            self._stack.addWidget(screen)
        self._stack.setCurrentWidget(screen)