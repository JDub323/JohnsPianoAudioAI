# Automatic Piano Music Transcription (Work in Progress)

This repository contains an **in-progress deep learning project** for **Automatic Music Transcription (AMT)**, focused on piano audio. The long-term goal is to build a system that converts raw audio into symbolic musical notation (e.g., MIDI) and eventually supports real-time transcription.

## Project Status
**Currently working on improving my model**, taking architectural inspiration from the *Mobile-AMT* paper.  

- The model currently uses a **stack of CNN layers followed by GRU layers**, loosely modeled after the Mobile-AMT design.  
- Training is performed on a personal laptop, where 100 epochs would take roughly 12 hours.  
- I’m actively addressing **memory leaks**, long training times, and stability issues during development.

## Corpus Generation
The corpus is built from the **MAESTRO dataset**, which provides paired piano audio and MIDI data.

- **Data Augmentation**: Includes several audio-domain transformations inspired by the augmentation strategies described in *Mobile-AMT*.

## Planned Work
- **Model Improvements**
  - Continue refining CNN/GRU architecture
  - Improve frame-level and onset prediction accuracy
  - Move toward real-time inference stability

- **Training Infrastructure**
  - Strengthen logging and experiment tracking
  - Integrate TensorBoard or Weights & Biases
  - Consider cloud-based training once local limits are reached

- **Evaluation**
  - Frame-level, onset-level, and note-level metrics
  - Compare performance against published AMT baselines

## Roadmap
1. Improve training stability and reduce memory-related issues  
2. Achieve reliable real-time prediction by end of the month (aspirational)  
3. Achieve *consistent* real-time prediction by end of the year (also aspirational)  
4. Expand team collaboration next semester to accelerate development  
5. Build a mobile app capable of displaying **Synthesia-style block note visualizations** as a semester-end goal

## Contributing
This project is still early-stage, but suggestions, insights, and collaboration are welcome.

## License
TBD (to be added once the project stabilizes)


