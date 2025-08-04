🚦 TrafficTelligence: Advanced Traffic Volume Estimation with Machine LearningOverviewThis project, developed as part of my internship, is an advanced machine learning system designed to predict traffic volume. Named TrafficTelligence, it uses historical data to forecast traffic with high accuracy, providing valuable insights for urban planning, dynamic traffic management, and commuter guidance. The solution is deployed as a web application using the Flask framework.
Project ArchitectureThe system follows a robust machine learning pipeline:
Data Preprocessing: The raw traffic volume.csv dataset is cleaned, missing values are handled, and new features are engineered from date and time information.
Model Training: A Random Forest Regressor model is trained on the prepared data.
Model Evaluation: The model's performance is validated on a dedicated test set.
Application Deployment: The trained model is saved and integrated into a Flask web application for real-time predictions.
Key Deliverables & Results
Model Performance:R-squared (R2) Score: 0.9696 (on the test set)
Root Mean Squared Error (RMSE): 346.51
Web Application: A functional web interface built with Flask allows users to input various parameters (date, time, weather, etc.) and receive a predicted traffic volume.
How to Run the ProjectTo run this project locally, follow these steps:
Clone the repository:git clone https://github.com/rejoice1574/Traffic_telligence.git
cd Traffic_telligence
Set up a virtual environment:python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
Install dependencies:pip install -r requirements.txt
Run the model training script to generate the necessary .pkl files:python train_model.py
Run the Flask application:flask --app Flask/app.py run
Open your web browser and navigate to http://127.0.0.1:5000/.Project Report & Demonstration VideoFor a detailed breakdown of the project, including a full technical analysis, methodology, and future work, please refer to the following documents.
[Link to your Word Document in Google Drive, Dropbox, etc.]
[Link to your Project Demonstration Video on YouTube, Vimeo, etc.]
