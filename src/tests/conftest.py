"""
conftest.py — Pin the Tesseract eng.traineddata model for integration tests.

Sets TESSDATA_PREFIX to the tessdata/ directory bundled with the test resources
so that OCR output is reproducible regardless of which system tessdata package
is installed. Without this, minor differences between tessdata versions cause
the same video frames to OCR differently, breaking timestamp assertions.
"""
import os

_TESSDATA_DIR = os.path.join(os.path.dirname(__file__), "resources", "tessdata")
os.environ["TESSDATA_PREFIX"] = _TESSDATA_DIR
