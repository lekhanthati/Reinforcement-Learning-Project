
# Policy Optimization for Financial Decision-Making

## ✨ Objective

This project is designed to assess the ability to handle a real-world machine 
learning problem from start to finish. We will take a public dataset, perform 
analysis, frame it for both supervised learning and reinforcement learning, build 
and train models, and—most importantly—critically analyze and compare their 
behaviors and outcomes. 

## 📂 Project Structure
```
├── main.py            # Main application 
├── Data_prep.py       # Data Pre-processing 
├── DL_model.py        # Deep Learning Model 
├── RL_model.py        # Reinforcement Learning Model
├── Requirements.txt   # Dependencies          
└── README.md         
```
## 🛠️ Tech Stack

| Technology    | Purpose                          |
|---------------|----------------------------------|
| **pandas**    | Data manipulation and analysis   |
| **numpy**     | Numerical computing operations   |
| **scikit-learn** | Machine learning utilities    |
| **tensorflow** | Deep learning model building    |
| **d3rlpy**    | Offline reinforcement learning   |

## 🔑 Environment Setup

Before running the app, make sure to set up your environment:

1. Create & activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r Requirements.txt
   ```
3. Place your dataset file accepted_2007_to_2018Q4.csv in the project root directory (same folder as main.py)
4. Run the app:
   ```bash
   python main.py
   ```
