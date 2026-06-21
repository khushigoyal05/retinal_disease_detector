from PySide6.QtWidgets import QMainWindow, QStackedWidget, QWidget
from PySide6.QtCore import Qt


class MainWindow(QMainWindow):
    """
    Root application window.
    QStackedWidget = only one screen visible at a time.
    All screen transitions go through show_screen().
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Retinal Disease Detector")
        self.setMinimumSize(800, 480)
        self.showFullScreen()  # fills the Pi touchscreen

        # Remove default menu/status bars for a clean full-screen look
        self.statusBar().hide()

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Global stylesheet — black/white medical theme
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
                border-radius: 8px;
                font-size: 15px;
                font-weight: 600;
                padding: 14px 24px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
            QPushButton:pressed {
                background-color: #000000;
            }
            QPushButton.outline {
                background-color: #FFFFFF;
                color: #111111;
                border: 2px solid #111111;
            }
            QPushButton.outline:hover {
                background-color: #F5F5F5;
            }
            QLabel {
                color: #111111;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #DDDDDD;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #111111;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 20px;
                height: 20px;
                margin: -8px 0;
                border-radius: 10px;
                background: #111111;
            }
        """)

    def show_screen(self, screen: QWidget) -> None:
        """Add screen to stack if new, then bring it to front."""
        if self._stack.indexOf(screen) == -1:
            self._stack.addWidget(screen)
        self._stack.setCurrentWidget(screen)