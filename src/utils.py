from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from collections import Counter

from collections import Counter

def majority_vote(chars):
    """Perform majority voting on a list of characters."""
    return Counter(chars).most_common(1)[0][0]

def determine_consensus(easyocr_text, pytesseract_text, trocr_text):
    """Combine the outputs of the OCR tools using a character-by-character majority voting."""
    # Find the length of the longest OCR output
    max_length = max(len(easyocr_text), len(pytesseract_text), len(trocr_text))
    
    combined_chars = []
    
    for i in range(max_length):
        # Get the character at the current index for each OCR output, or an empty string if the index is out of range
        easy_char = easyocr_text[i] if i < len(easyocr_text) else ""
        pytes_char = pytesseract_text[i] if i < len(pytesseract_text) else ""
        trocr_char = trocr_text[i] if i < len(trocr_text) else ""

        chars = [easy_char, pytes_char, trocr_char]
        
        # Perform majority voting on the characters
        most_common_char = majority_vote(chars)
        
        # Add the most common character to the combined output
        combined_chars.append(most_common_char)
    
    return "".join(combined_chars)
