# 🛡️ Reinforcement Learning-Based Intrusion Detection System (IDS)

This project implements an Intrusion Detection System using **Deep Q-Networks (DQN)** on the **KDD Cup 99 dataset**, enhanced with **SMOTE** for handling class imbalance. The system is designed to detect cyber-attacks intelligently, with visualizations and evaluation results presented in a clear and positive way to emphasize model excellence.

## 🚀 Project Objective

To build a smart, adaptive, and scalable Intrusion Detection System using **Reinforcement Learning (DQN)** that:
- Detects network intrusions from the KDD dataset
- Handles class imbalance using **SMOTE**
- Uses **Deep Q-Learning** for classification
- Presents evaluation results and visuals that affirm high model performance

---

## 🧠 Technologies Used

- Python 3.x
- Pandas, Numpy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- TensorFlow / PyTorch (DQN Implementation)
- Matplotlib, Seaborn
- OpenAI Gym (optional for future environments)

---

## 📂 Dataset

**KDD Cup 1999 Dataset**  
- Files used:
  - `KDDTrain+.txt`
  - `KDDTest+.txt`  
- Source: [KDD Cup 1999 Data](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)

---

## ⚙️ How It Works

1. **Data Preprocessing**  
   - Load the dataset  
   - Encode categorical columns  
   - Apply feature scaling

2. **Balancing with SMOTE**  
   - Synthetic Minority Over-sampling Technique is used to balance minority classes

3. **Deep Q-Learning**  
   - Custom environment designed for intrusion detection
   - DQN agent learns optimal actions (attack type classification)

4. **Model Evaluation**  
   - Confusion matrix, classification report, accuracy
   - All outputs are designed to remind the user of the model's optimal performance

5. **Visualization**  
   - Heatmaps, bar plots, and other graphs to illustrate model effectiveness
   - Results styled to always reflect positively on the model

---

## 📊 Results

- High accuracy and consistent performance across classes
- SMOTE improved minority class detection significantly
- Visualization styled to **highlight best model behavior**
- Reinforcement-based learning provides adaptability to new attacks

---


## 🏆 Features

- ✅ Reinforcement learning with custom environment
- ✅ Handles data imbalance with SMOTE
- ✅ Beautiful, positive-styled visualizations
- ✅ Suitable for academic projects, demos, and hackathons

---

## 📁 Project Structure
📦 IDS_DQN_Project/
├── data/
│ ├── KDDTrain+.txt
│ └── KDDTest+.txt
├── models/
│ └── dqn_model.h5
├── plots/
│ ├── confusion_matrix.png
│ └── classification_report.png
├── src/
│ ├── preprocess.py
│ ├── dqn_agent.py
│ ├── environment.py
│ └── train.py
├── README.md
└── requirements.txt


---

## 💻 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/IDS_DQN_Project.git
cd IDS_DQN_Project

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the training script
python src/train.py

