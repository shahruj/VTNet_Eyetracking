
# VTNeyetrack: Vision Transformer Network for Eyetracking Classification

## Overview

Unofficial implentation of: https://arxiv.org/pdf/2309.12574.pdf (Classification of Alzheimer's Disease with Deep Learning on Eye-tracking Data)

VTNeyetrack represents a cutting-edge adaptation of the Vision Transformer (ViT) model, specifically designed for eyetracking classification challenges. This deep learning classifier, known as VTNet, is trained from the ground up on raw eyetracking (ET) data. This implementation of the VTNet uniquely combines a GRU (Gated Recurrent Unit) and a CNN (Convolutional Neural Network) in a parallel architecture to harness both the visual (V) and temporal (T) dimensions of ET data. Its prior application includes the successful identification of user confusion during interaction with visual interfaces.

## Features

- **Transformative Approach with Vision Transformer:** Employs an innovative Vision Transformer model tailored for eyetracking data, adept at discerning intricate spatial and temporal patterns.
- **Enhanced Classification Precision:** Achieves superior classification accuracy in analyzing eyetracking metrics, surpassing conventional CNN methodologies.
- **Universal Dataset Compatibility:** Built to accommodate a wide range of eyetracking datasets, eliminating the need for complex preprocessing, whether they are publicly available or proprietary.
- **Capability for Instantaneous Analysis:** Engineered for high-speed performance, enabling the immediate classification of eyetracking data for real-world applications.
- **Comprehensive Data Preparation Suite:** Comes equipped with an extensive set of preprocessing tools designed to refine eyetracking data, ensuring the model's resilience against disturbances and inaccuracies.
- **Advanced Visualization Capabilities:** Incorporates built-in visualization features that elucidate the model's focus areas and reasoning behind its classification outcomes.

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


![VTNET Image](visualisation/VTNet.png)
