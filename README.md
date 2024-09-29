---
title: GOT OCR
emoji: 👀
colorFrom: green
colorTo: indigo
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
license: mit
---

# Image Text OCR using GOT (General OCR Theory)

This Streamlit application provides an interface for Optical Character Recognition (OCR) using the GOT (General OCR Theory) model. Users can upload images, and the application will extract text from them. Additionally, it includes a search feature to highlight specific words or phrases in the extracted text.

## Demo

Try out the live demo on huggingface: [GOT-OCR-Space](https://huggingface.co/spaces/akhil-vaidya/GOT-OCR)

## Setup

1. Clone this repository to your local machine.
2. Ensure you have Python 3.7+ installed.
3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application locally:

```
streamlit run app.py
```

Then, open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Features

- Image upload: Supports JPG, PNG, and JPEG formats.
- Text extraction: Uses the GOT model to extract text from uploaded images.
- Search functionality: Allows users to search for specific words or phrases in the extracted text, highlighting the matches.

## Model Information

This application uses the CPU version of the GOT (General OCR Theory) model. The model is loaded from the Hugging Face model hub:

- Model: `srimanth-d/GOT_CPU`
- Tokenizer: `srimanth-d/GOT_CPU`

While the code includes functions for GPU model initialization, the deployed version uses the CPU model due to limitations in the Hugging Face deployment environment. This may result in slower processing times but ensures compatibility across different deployment scenarios.

## Note on Performance

The use of the CPU model might lead to slower processing times compared to a GPU-enabled version. However, this trade-off ensures that the application can run in environments without GPU support, such as the Hugging Face deployment platform.

## Additional Information

- The application creates a directory named `images` to temporarily store uploaded images.
- The search functionality is case-insensitive and highlights all occurrences of the search term in the extracted text.