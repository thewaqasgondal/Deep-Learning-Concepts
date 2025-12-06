# Deep-Learning-Concepts

This repository is a comprehensive collection of deep learning experiments and concepts, covering everything from basic fundamentals to advanced techniques. It serves as a practical guide and codebase for learning and implementing deep learning models, algorithms, and applications.

## Overview

The repository is organized to provide a structured learning path:

- **Basic Concepts**: Introduction to neural networks, activation functions, loss functions, and optimization algorithms.
- **Intermediate Topics**: Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and common architectures.
- **Advanced Experiments**: Cutting-edge techniques like transformers, generative models, reinforcement learning, and custom implementations.

## Repository Structure

- `00_test.ipynb`: Jupyter notebook for initial testing
- `01_test_env.py`: Python script to verify environment setup
- `02_digit_exp.py`: Analysis and exploration of the sklearn digits dataset
- `03_digit_my_data.py`: Analysis of custom Kaggle digits dataset (reads from zip file)
- `requirements.txt`: List of required Python packages
- `dataset/`: Directory containing dataset files (e.g., zipped datasets)
- `output/`: Directory for storing analysis results, plots, and visualizations

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Required libraries: NumPy, Pandas, Matplotlib, TensorFlow/PyTorch, Scikit-learn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/thewaqasgondal/Deep-Learning-Concepts.git
   cd Deep-Learning-Concepts
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Run the environment verification scripts:
   ```bash
   python 01_test_env.py
   ```

2. Explore dataset analyses:
   ```bash
   python 02_digit_exp.py  # sklearn digits
   python 03_digit_my_data.py  # Kaggle digits from zip
   ```

3. Check results in the `output/` directory for generated plots and visualizations.

Each experiment is self-contained and includes data loading, analysis, and visualization. Results are automatically saved to the `output/` folder for easy review.

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
