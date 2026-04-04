# 🎨 Fashion AI: Your Personal AI Stylist

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-05998b.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00.svg)](https://wwww.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Fashion AI** is a state-of-the-art recommendation system that uses Deep Learning to analyze your wardrobe and suggest perfectly matched outfits. Whether you're dressing for a party, a date, or the office, Fashion AI helps you look your best by understanding style, color, and texture.

---

##  Key Features

- **Wardrobe Management**: Upload your tops and bottoms through a modern, intuitive interface.
-  **AI-Powered Analysis**: Uses pre-trained Convolutional Neural Networks (ResNet50/VGG16) to "see" your clothes.
-  **Smart Recommendations**: Finds the best matches based on visual similarity and occasion-specific logic.
-  **One-Click Launch**: A single command starts the backend and opens the frontend in your browser.
-  **Responsive Design**: A sleek, dark-mode dashboard that works on any device.

---

## Technology Stack

- **Backend core**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Deep Learning**: [TensorFlow](https://wwww.tensorflow.org/) & [Keras](https://keras.io/) (ResNet50, VGG16)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (Cosine Similarity)
- **Image Processing**: [OpenCV](https://opencv.org/), [Pillow](https://python-pillow.org/)
- **Frontend**: HTML5, Vanilla CSS3, JavaScript
- **Development**: [Uvicorn](https://www.uvicorn.org/), [NumPy](https://numpy.org/)

---

##  Getting Started

### 1. Prerequisites

- Python 3.8 or higher.
- A virtual environment is recommended.

### 2. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/samridh-111/fashion_ai.git
   cd fashion_ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements_new.txt
   ```

### 3. Running the Application

Simply run the launcher script:
```bash
python run.py
```

This will:
1. Check for all necessary dependencies.
2. Initialize the ML models and start the FastAPI backend on `http://localhost:8000`.
3. Automatically open the modern dashboard in your default web browser.

---

## 📁 Project Structure

| File | Description |
| :--- | :--- |
| `run.py` | Main entry point; starts server and opens browser. |
| `backend.py` | FastAPI server handling API requests and storage. |
| `ml_features.py` | Handles image preprocessing and feature extraction using CNNs. |
| `recommendation_engine.py` | Contains the logic for matching clothes and scoring outfits. |
| `frontend.html` | The user-facing dashboard for wardrobe management. |
| `extract_features.py` | Utility for batch-processing entire datasets. |

---

##  How it Works

1. **Feature Extraction**: When you upload an image, it's passed through a Deep Learning model (ResNet50). The model converts the visual information into a unique "fingerprint" called a **feature vector** (a list of 2048 numbers).
2. **Similarity Matching**: To find a match for a top, the system compares its feature vector against all stored bottoms using **Cosine Similarity**.
3. **Occasion Filtering**: The engine applies additional logic to prioritize certain styles depending on whether you've selected "Casual", "Office", "Party", or "Date".
4. **Final Recommendation**: The highest-scoring combinations are returned and displayed in the frontend.

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

*Made for the Fashion Community.*
