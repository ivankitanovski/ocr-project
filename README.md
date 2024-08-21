# OCR Challenge Project

## Overview

This project extracts and analyzes text from images using multiple OCR tools (EasyOCR, PyTesseract, and TrOCR). The goal is to predict the correct text output and store the results in a structured JSON format.

The current approach uses a consensus method on a character by character basis. It processes the predicted strings from the three models one character at a time and selects the most common character at `i`-th positition for the final prediction. This process goes for `i` from `0` to the length of the longest prediction.

There could be other approaches, where you could look on a word level or run another analysis on top of that. You could also test the models independetly an assign weights to them based on their validation performance, or maybe use their confidence score etc.

## Setup

Before running the script, ensure you have the following installed:

- Python 3.x
- Tesseract OCR engine

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

To run the OCR process just execute:

```bash
python src/main.py --images PATH_TO_IMAGES_DIR --output PATH_TO_OUTPUT_FILE
```

- `PATH_TO_IMAGES_DIR` defaults to `data/images`
- `PATH_TO_OUTPUT_FILE` defaults to `output/results.json`

_Note_: Make sure to set the path to the tesseract binary in the `TESSERACT_CMD` variable in the `.env` file, in case Tesseract is not set in your path variables. It can happen on Windows machines.

## Project Structure

- `data/`: Contains the image dataset (`selected_images.zip`) and extracted images under the (`images`).
  - Make sure you unzip the dataset in this folder.
- `output/`: Stores the final JSON output and application logs
- `src/`: Python scripts for the application running the OCR processing, analysis, and JSON generation.
  - `main.py`: The main script that runs and executes the OCR process
  - `models.py`: A package abstracting the models being used in the OCR process.
  - `utils.py`: Contains the logic to determine the consesus of the separate models.
- The root folder also contains configuration files, such as `logging.conf` to configure the logging. The `.env` file is used to specify the tesseract binary path (might need to specify it for Windows).
