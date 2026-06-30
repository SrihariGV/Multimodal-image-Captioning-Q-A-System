# Multimodal Image Captioning + Q&A System

## Project Overview

The **Multimodal Image Captioning + Q&A System** is an Artificial Intelligence project that combines Computer Vision and Natural Language Processing (NLP). The system generates captions for uploaded images and answers questions related to the image content.

## Objectives

* Generate captions for images automatically.
* Answer user questions based on image content.
* Integrate image understanding with natural language processing.

## Features

* Image Upload
* Image Feature Extraction
* Automatic Caption Generation
* Visual Question Answering (VQA)
* Simple User Interface

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Streamlit
* VGG16 (Feature Extraction)

## Project Structure

```
Multimodal_Image_Captioning_QA/
│
├── dataset/
│   ├── images/
│   ├── captions.csv
│   ├── train.csv
│   ├── test.csv
│   └── val.csv
│
├── models/
│   ├── caption_model.h5
│   ├── tokenizer.pkl
│   └── image_features.pkl
│
├── feature_extraction.py
├── dataset_preparation.py
├── train_model.py
├── image_caption.py
├── visual_qa.py
├── predict.py
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

## Dataset

* Flickr8k Dataset (Image Captioning)
* VQA v2 Dataset (Visual Question Answering)

## Installation

1. Clone the repository.
2. Install the required libraries:

   ```
   pip install -r requirements.txt
   ```
3. Run the application:

   ```
   streamlit run app.py
   ```

## Expected Output

* Upload an image.
* Generate a descriptive caption.
* Answer questions related to the uploaded image.

## Future Enhancements

* Improve caption accuracy using Transformer models.
* Support multiple languages.
* Add real-time webcam image captioning.
* Deploy the application to the cloud.

## Author

**Srihari G.V.**
