
# VTNeyetrack: Vision Transformer Network for Eyetracking Classification

## Overview

Unofficial implentation of: https://arxiv.org/pdf/2309.12574.pdf (Classification of Alzheimer's Disease with Deep Learning on Eye-tracking Data)

VTNeyetrack is an innovative implementation of the Vision Transformer (ViT) architecture tailored specifically for eyetracking classification tasks. Our project leverages the powerful capabilities of transformer models, traditionally used in natural language processing (NLP), and adapts them for the nuanced requirements of eyetracking data analysis. This approach allows for exceptional accuracy in classifying various eyetracking metrics, including gaze direction, fixation duration, and blink rate, among others.

## Features

- **Transformer-Based Architecture**: Utilizes a customized Vision Transformer model to process eyetracking data, capturing complex spatial relationships and temporal dynamics.
- **High Accuracy Classification**: Designed for high performance in classifying eyetracking metrics, improving upon traditional CNN-based approaches.
- **Dataset Agnostic**: Flexible to work with various eyetracking datasets, both public and proprietary, without requiring extensive preprocessing.
- **Real-Time Analysis Capability**: Optimized for efficiency, allowing for real-time classification of eyetracking data in practical applications.
- **Extensive Preprocessing Toolkit**: Includes tools for cleaning and preparing eyetracking data, making the model robust to noise and artifacts.
- **Visualization Tools**: Features integrated visualization tools to interpret the model's attention mechanisms and classification decisions.

## Installation

Clone this repository to your local machine:

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the VTNeyetrack model on your eyetracking data, follow these steps:

1. Prepare your data according to the guidelines provided in the `data_preparation` folder.
2. Adjust the configuration settings in `config.py` to match your dataset and desired model parameters.
3. Train the model using:

```bash
python train.py --config config.py
```

4. For classification, use:

```bash
python classify.py --input your_data_file.csv --output predictions.csv
```

## Contributing

We welcome contributions from the community! If you have improvements or bug fixes, please submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

For any inquiries or further discussion, feel free to contact us through GitHub issues or directly at our email addresses provided in the repository.

---

This project is at the cutting edge of applying transformer models to the field of eyetracking, pushing forward the capabilities in analyzing and interpreting complex eyetracking data. Join us in advancing this exciting area of research and application!
