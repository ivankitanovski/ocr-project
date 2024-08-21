import os
import easyocr
import pytesseract
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class OCRModel:
    def predict(self, image_path):
        raise NotImplementedError
        
class EasyOCRModel(OCRModel):
    def __init__(self, ):
        super().__init__()
        self.model = easyocr.Reader(['en']) 
    
    def predict(self, image_path):
        result = self.model.readtext(image_path, detail=0)
        return " ".join(result).strip()

class PyTesseractModel(OCRModel):
    def __init__(self, ):
        super().__init__()
        # set the path to the tesseract executable, if not set in the environment. It'll read it from the .env file
        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    def predict(self, image_path):
        return pytesseract.image_to_string(Image.open(image_path)).strip()

class TrOCRModel(OCRModel):
    def __init__(self, ):
        super().__init__()
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()