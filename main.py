import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PIL import Image, ImageFilter
import numpy as np

from ui.main_window import MainWindow
from ui.home_screen import HomeScreen
from ui.camera_screen import CameraScreen
from ui.preview_screen import PreviewScreen
from ui.results_screen import ResultsScreen
from ui.inference_worker import InferenceWorker
from core.pipeline import InferencePipeline
from services.file_service import FileService
from services.inference_service import TFLiteInferenceEngine
from utils.image_utils import compute_sharpness_score
import config


def _simulate_capture(preview_screen: PreviewScreen) -> None:
    config.CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    base = Image.new("RGB", (224, 224), color=(180, 100, 60))
    paths, scores = [], []
    for i, blur in enumerate([0.5, 2.0, 4.0]):
        img = base.filter(ImageFilter.GaussianBlur(radius=blur))
        p = config.CAPTURE_DIR / f"test_frame_{i}.jpg"
        img.save(str(p))
        scores.append(compute_sharpness_score(np.array(img)))
        paths.append(p)
    preview_screen.load_frames(paths, scores)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()

    home    = HomeScreen()
    camera  = CameraScreen()
    preview = PreviewScreen()
    results = ResultsScreen()

    # Inference engine — will fail gracefully if no model exists yet
    engine = TFLiteInferenceEngine()
    try:
        engine.load()
    except FileNotFoundError:
        print("No model found — inference will fail until model is trained")
    file_source = FileService()

    # Navigation
    home.capture_requested.connect(lambda: window.show_screen(camera))
    def on_upload():
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            window,
            "Select Retinal Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return  # user cancelled

        try:
            file_source.select(Path(path))
            pipeline = InferencePipeline(
                image_source=file_source,
                inference_engine=engine
            )
            worker = InferenceWorker(pipeline)
            window._worker = worker

            sharpness = compute_sharpness_score(
                np.array(Image.open(path).convert("RGB"))
            )

            def on_done(result):
                results.load_results(result, sharpness)
                window._worker = None

            def on_fail(error):
                print(f"Inference failed: {error}")
                window._worker = None

            worker.finished.connect(on_done)
            worker.failed.connect(on_fail)
            worker.start()
            window.show_screen(results)

        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(window, "Upload failed", str(e))

    home.upload_requested.connect(on_upload)
    camera.back_requested.connect(lambda: window.show_screen(home))
    camera.capture_requested.connect(
        lambda: [_simulate_capture(preview), window.show_screen(preview)]
    )
    preview.retake_requested.connect(lambda: window.show_screen(camera))
    results.back_requested.connect(lambda: window.show_screen(preview))
    results.new_scan_requested.connect(lambda: window.show_screen(home))

    def on_analyze(image_path: Path):
        """Run inference in background, show results when done."""
        window.show_screen(results)
        file_source.select(image_path)
        pipeline = InferencePipeline(
            image_source=file_source,
            inference_engine=engine
        )

        sharpness = compute_sharpness_score(
            np.array(Image.open(image_path).convert("RGB"))
        )

        worker = InferenceWorker(pipeline)

        # Store on window so Python doesn't garbage collect it
        # while the thread is still running
        window._worker = worker

        def on_done(result):
            results.load_results(result, sharpness)
            window._worker = None  # safe to release now

        def on_fail(error):
            print(f"Inference failed: {error}")
            window._worker = None

        worker.finished.connect(on_done)
        worker.failed.connect(on_fail)
        worker.start()

        def on_done(result):
            results.load_results(result, sharpness)

        def on_fail(error):
            print(f"Inference failed: {error}")

        worker.finished.connect(on_done)
        worker.failed.connect(on_fail)
        worker.start()

    preview.analyze_requested.connect(on_analyze)

    window.show_screen(home)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()