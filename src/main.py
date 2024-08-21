import argparse
import logging
import logging.config
import os
import json
from dotenv import load_dotenv
from utils import determine_consensus
from models import EasyOCRModel, PyTesseractModel, TrOCRModel

# load env file
load_dotenv()

# Load logging configuration
logging.config.fileConfig(os.path.join(os.path.dirname(__file__), '..', 'logging.conf'))

# Get the root logger
logger = logging.getLogger()

# Define default paths
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "data/images")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "output/results.json")

def process_images(image_dir):
    results = []

    logging.info(f"Loading models...")
    # Load the models
    easyocr_model = EasyOCRModel()
    pytesseract_model = PyTesseractModel()
    trocr_model = TrOCRModel()
    
    logging.info(f"Processing images from {image_dir}...")
    # Iterate over each image in the directory
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        if os.path.isfile(image_path):
            logging.info(f"Processing {image_name}...")
            try:
                # Run OCR using the three tools
                easyocr_text = easyocr_model.predict(image_path)
                pytesseract_text = pytesseract_model.predict(image_path)
                trocr_text = trocr_model.predict(image_path)
                logging.info(f"Processed image {image_name}, EasyOCR: {easyocr_text}, Pytesseract: {pytesseract_text}, TrOCR: {trocr_text}")

                # Determine the most accurate text
                final_text = determine_consensus(easyocr_text, pytesseract_text, trocr_text)
                logging.info(f"Final text for {image_name}: {final_text}")

                # Append result to the list
                results.append({
                    "image_name": image_name,
                    "text": final_text
                })
            except Exception as e:
                logging.error(f"Error processing {image_name}: {e}")
    
    return results

def save_results(results, output_file):
    # Save the results to a JSON file
    logging.info(f"Saving results to {args.output}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def main(images_dir=IMAGE_DIR, output_file=OUTPUT_FILE):
    """ Main function to process images and save the OCR results to a JSON file. """
    logging.info("Starting OCR processing...")
    
    # Process images and get the OCR results
    ocr_results = process_images(images_dir)
    
    # Save the results to a JSON file
    save_results(ocr_results, output_file)

    logging.info("OCR processing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR on a folder with images and store the output in designated JSON file.")
    parser.add_argument('--images', type=str, help='The folder where the images are stored.', default=IMAGE_DIR)
    parser.add_argument('--output', type=str, help='The file where the output will be stored.', default=OUTPUT_FILE)
    args = parser.parse_args()

    main(args.images, args.output)    
