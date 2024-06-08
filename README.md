# Unified Pre-Trained Vision-Language BLIP For Medical VQA

# Overview

This repository focuses on leveraging a unified vision-language pre-trained model for fine-tuning on medical question answering datasets. The core model architecture is based on the BLIP (Bootstrapped Language Image Pretraining) framework, which integrates visual and language information to generate accurate responses to medical-related queries.

# Contents 
   * Introduction
   * Installation
   * BLIP Model Architecture
   * Dataset Preparation
   * Model Fine-Tuning
   * Text Generation From Medical Image
   * Results
   * Acknowledgements

# Introduction 

In this project, we discuss a Vision Transformer (ViT) as the image encoder, which processes input images through a series of transformer layers, splitting them into fixed-sized patches and generating visual features. The text encoder, also based on the transformer architecture, encodes textual inputs such as questions or captions into textual features, tokenizing and embedding the text before passing it through transformer layers to capture linguistic context.

BLIP is pre-trained on large-scale datasets using tasks like masked language modeling, image-text matching, and contrastive learning. This pre-training helps the model learn a robust multi-modal representation, which can then be fine-tuned on specific downstream tasks such as visual question answering. Depending on the task, BLIP can have different output heads attached to the fused representation, such as a classification head for multiple-choice questions or a generation head for creating text captions based on the image.



# Installation 

# BLIP Model Architecture 
  * ### Contrasitive Pre-Training
  * ### Caption Filtering and Generation

# Dataset Preparation 

# Model Fine-Tuning 

# Text Generation From Medical Image 


