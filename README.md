

# 🚦 TrafficTelligence: Advanced Traffic Volume Estimation with Machine Learning

## 📌 Overview

**TrafficTelligence** is a machine learning-based web application developed during my internship to estimate traffic volume with high accuracy. It uses historical traffic data to make real-time predictions and offers insights beneficial for:

* Urban planning
* Dynamic traffic management
* Commuter guidance

The solution is deployed using the **Flask** framework for easy interaction and visualization.

---

## 🏗️ Project Architecture

### 🔹 Data Preprocessing

* Loaded and cleaned `traffic_volume.csv`
* Handled missing values
* Engineered new features from date and time

### 🔹 Model Training

* Trained using **Random Forest Regressor**

### 🔹 Model Evaluation

* **R² Score**: `0.9696` (on test set)
* **RMSE**: `346.51`

### 🔹 Deployment

* Trained model saved as `.pkl`
* Integrated with a **Flask** web interface for real-time predictions

---

## 💻 Web Application

Users can input parameters such as:

* Date and Time
* Weather Conditions

The app returns a predicted traffic volume.

---

## 🚀 How to Run the Project

### 🧬 Clone the Repository

```bash
git clone https://github.com/rejoice1574/Traffic_telligence.git
cd Traffic_telligence
```

### 🐍 Set Up Virtual Environment

```bash
python -m venv venv
```

#### ▶️ Activate It:

* **Windows**:

  ```bash
  .\venv\Scripts\activate
  ```

* **macOS/Linux**:

  ```bash
  source venv/bin/activate
  ```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🧠 Train the Model

```bash
python train_model.py
```

### 🌐 Run the Web Application

```bash
flask --app Flask/app.py run
```

Then open your browser and visit:
👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 📄 Documentation & Demo

*  Word Document : https://drive.google.com/file/d/1Ant_K_3kCGPoR2vLjrWtoQSUHEf2GHnv/view?usp=drivesdk
* Demo video : https://drive.google.com/file/d/1Az0N37PDOMgD3hn98hb-zChXsus94CjU/view?usp=drivesdk

---


