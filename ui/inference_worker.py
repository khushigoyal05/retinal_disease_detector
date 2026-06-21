from pathlib import Path
from PySide6.QtCore import QThread, Signal
from core.pipeline import InferencePipeline
from core.models import PredictionResult


class InferenceWorker(QThread):
    """
    Runs InferencePipeline in a background thread.
    Never touch UI from here — only emit signals.

    Signals:
        finished(PredictionResult): inference completed successfully
        failed(str):                something went wrong — error message
    """
    finished = Signal(PredictionResult)
    failed = Signal(str)

    def __init__(self, pipeline: InferencePipeline, parent=None):
        super().__init__(parent)
        self._pipeline = pipeline

    def run(self) -> None:
        """Qt calls this automatically when .start() is called."""
        try:
            result = self._pipeline.run()
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))