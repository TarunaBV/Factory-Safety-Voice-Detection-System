# Factory-Safety-Voice-Detection-System

## Problem Statement:

In heavy industrial and manufacturing environments, workers often face safety-critical situations where immediate communication is required to prevent accidents or operational hazards. However, existing emergency reporting mechanisms such as manual switches, handheld devices, or cloud-dependent voice assistants are often unreliable due to extreme background noise, limited mobility, and network constraints.

There is a need for a real-time, noise-robust, hands-free safety reporting system capable of accurately detecting predefined emergency voice commands in high-noise industrial conditions. The system must operate with low latency, high reliability, and offline capability to ensure timely safety alerts and monitoring.

This project aims to design and develop a software-based industrial voice safety detection system that continuously processes streaming audio, identifies safety-critical voice commands under noisy conditions, and provides real-time alert notifications along with analytics support for safety monitoring.

## Architectural Diagram : 

![Architecture](assets/Architecture_dark.png)

## Project Structure

```
factory-safety-voice-detection/
│
├── assets/
│   ├── audio_utils.py
│   ├── noise_reduction.py
│   ├── feature_utils.py
│   ├── db_utils.py
│   ├── config.py
│   └── logger.py
│
├── program/
│   ├── audio_input.py
│   ├── vad.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── keyword_spotting.py
│   ├── confidence_gate.py
│   ├── alert_system.py
│   └── main.py
│
├── models/
│   ├── ds_cnn_model.pth
│   ├── labels.txt
│   └── model_loader.py
│
├── data/
│   ├── raw_audio/
│   ├── processed_audio/
│   ├── features/
│   └── logs/
│
├── database/
│   ├── events.db
│   └── schema.sql
│
├── dashboard/
│   ├── app.py
│   ├── components/
│   └── static/
│
├── templates/
│   ├── index.html
│   └── dashboard.html
│
├── tests/
│   ├── test_vad.py
│   ├── test_model.py
│   └── test_pipeline.py
│
├── scripts/
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── data_augmentation.py
│ 
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```