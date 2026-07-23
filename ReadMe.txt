Federated Learning with Personalization and EWC for New Client Adaptation

1. Overview:
This repository contains the source code for the paper: "Privacy-Preserving WiFi-based Gesture Recognition Using Personalized Federated Learning"
The main contributions of this work are as follows:
- A metadata-driven harmonization step proposed to sort gesture labels across recording sessions
- To address class-level imbalance (different gesture subsets were recorded across sessions), and sample-level imbalance (uneven user participation across rooms and recording days), a data-driven severity assessment estimates users' class coverage, label overlap, and KL divergence between local train and validation distributions and directs users to employ personalized layers or not
- Dynamic client participantions would be achieved by Elastic Weight Consolidation (EWC)-anchored adaptation through a federated approximation of Fisher Information


2. Requirements
- Python 3.9+
- TensorFlow 2.x (Preferred 2.10 for Windows GPU support)
- NumPy
- SciPy
- scikit-learn
- psutil

3. Dataset
This code uses the Widar 3.0 dataset
- Download from: https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset
- Check for the filename "BVP" which contains 14 subfolders named after the date the gestures are collected. For example "20181109-VS", "20181121-VS"
- Check for Dataset 2: "20181112-VS" and remove from the BVP folder
- Place the cleaned BVP folder in your working directory and rename it `BVPwithout2/`


4. How to run
- Step 1: Run the data loading section from the submitted Jupyter Notebook
- Step 2: Run experiments
  Three cases are provided:
  Case 1: Samples with same classes in the train/val/test sets as initial users → Samples with class imbalance as dynamic clients
  Case 2: Samples with class imbalance as initial users → Samples with same classes in the train/val/test sets as dynamic clients
  Case 3: Mixed initial clients → Mixed clients join
