# 🎹 Automatic Piano Music Transcription (Work in Progress)

This repository contains an **in-progress deep learning project** for **Automatic Music Transcription (AMT)**, with a focus on piano recordings. The goal is to build a model that converts raw audio into symbolic musical notation (e.g., MIDI).  

## Project Status
🚧 **Currently in development**  
- ✅ Corpus generation implemented  
- ✅ High-level pseudocode for training and inference  
- 🔄 Model architecture (deep dive in progress)  
- 🔄 Logging, experiment tracking, and training pipeline  

---

## 📀 Corpus Generation
The corpus is built from the [**MAESTRO dataset**](https://magenta.tensorflow.org/datasets/maestro), which contains paired piano audio and MIDI files.  

- **Data Augmentation**: Applied audio transformations inspired by the augmentation pipeline in [Audio-AMT](https://ieeexplore.ieee.org/document/10715008).  

---

## 🧩 Planned Work
- **Model Architecture**:  
  - Exploring U-Net style encoders combined with sequence models (e.g., Transformers)  
  - Investigating recent AMT model designs for piano transcription  

- **Training & Logging**:  
  - Implement robust logging for losses, metrics, and visualizations  
  - Integrate with tools like TensorBoard or Weights & Biases  

- **Evaluation**:  
  - Frame-level, note-level, and onset-offset metrics  
  - Comparisons against baseline AMT models  

---

## 📌 Roadmap
1. Finalize model design  
2. Implement training & evaluation loops  
3. Add experiment tracking & logging  
4. Release pretrained checkpoints and demo notebooks  

---

## 🤝 Contributing
This project is still in an early phase, but feedback and suggestions are welcome!  

---

## 📜 License
TBD (to be added once the project stabilizes).

