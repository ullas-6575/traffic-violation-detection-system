"""
plate_module
TVS-10: License plate localization + OCR, as a self-contained package.
"""

from .pipeline import PlateRecognitionPipeline, PlateRecognitionResult
from .detector import PlateLocator, OpenCVPlateLocator, YOLOPlateLocator
from .ocr_reader import PlateOCRReader, OCRResult

__all__ = [
    "PlateRecognitionPipeline",
    "PlateRecognitionResult",
    "PlateLocator",
    "OpenCVPlateLocator",
    "YOLOPlateLocator",
    "PlateOCRReader",
    "OCRResult",
]
