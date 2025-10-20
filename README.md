# 🏍️ Two-Wheeler Rider Intention Prediction

This project uses **Machine Learning** to predict the **riding intention of a two-wheeler driver** (e.g., left turn, right lane change, straight, stop, etc.) based on extracted **VGG16 deep learning features** from rider movement data.

The web interface is built with **Streamlit**, allowing users to upload `.npy` feature files and receive real-time predictions of rider intentions.

---

## 🚀 Features
- Upload `.npy` files containing extracted VGG16 features  
- Automatic preprocessing and feature pooling  
- Machine Learning–based classification of riding intentions  
- Interactive and lightweight Streamlit web interface  
- Easily deployable on **Streamlit Cloud**  

---

## 🧠 Model Information
The prediction model was trained using **scikit-learn** and **XGBoost**, with optional feature scaling applied via **StandardScaler**.  
Classes predicted by the model include:

| Numeric Label | Predicted Intention      |
|----------------|--------------------------|
| 0              | Left Lane Change         |
| 1              | Left Turn                |
| 2              | Right Lane Change        |
| 3              | Right Turn               |
| 4              | Slow-Stop                |
| 5              | Straight                 |

---

## 🧩 Installation and Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/two-wheeler-intention-prediction.git
cd two-wheeler-intention-prediction
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app
```bash
streamlit run app.py

## 🧾 License
This project is open-source and available under the **MIT License**.
