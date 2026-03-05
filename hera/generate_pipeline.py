import logging
import os
import threading

from hera import config

logger = logging.getLogger("hera.generate_pipeline")


class GeneratePipeline:
    def __init__(self):
        self.animator = None
        self._thread = None
        self._cancel_event = threading.Event()
        self._generating = False
        self._error = None
        self._output_path = None
        self._progress = (None, None)

    @property
    def is_generating(self):
        return self._generating

    @property
    def error(self):
        return self._error

    @property
    def output_path(self):
        return self._output_path

    @property
    def progress(self):
        return self._progress

    def load_models(self, progress_callback=None):
        from hera.processors.portrait_animator import PortraitAnimator

        self.animator = PortraitAnimator()

        # Check that the pretrained weights exist
        pretrained = config.LIVEPORTRAIT_PRETRAINED_WEIGHTS
        expected = os.path.join(pretrained, "liveportrait", "base_models", "appearance_feature_extractor.pth")
        if not os.path.exists(expected):
            raise FileNotFoundError(
                f"LivePortrait models not found at {pretrained}\n"
                "Run: python setup_models.py"
            )

        self.animator.load(progress_callback=progress_callback)

    def generate(self, source_image, driving_video, output_dir=None, progress_callback=None, done_callback=None):
        if self._generating:
            return
        self._cancel_event.clear()
        self._error = None
        self._output_path = None
        self._progress = (None, None)
        self._generating = True

        output_dir = output_dir or config.GENERATE_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        def _run():
            try:
                def _progress(status, detail):
                    self._progress = (status, detail)
                    if progress_callback:
                        progress_callback(status, detail)

                result = self.animator.animate_frames(
                    source_image,
                    driving_video,
                    output_dir=output_dir,
                    progress_callback=_progress,
                    cancel_event=self._cancel_event,
                )

                if self._cancel_event.is_set():
                    self._output_path = None
                else:
                    self._output_path = result

            except Exception as e:
                logger.error(f"Generation failed: {e}")
                self._error = str(e)
            finally:
                self._generating = False
                if done_callback:
                    done_callback()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def cancel(self):
        if self._generating:
            self._cancel_event.set()
