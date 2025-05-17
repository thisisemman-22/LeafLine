# LeafLine: Plant Disease Classification Web App

LeafLine is a web application that uses deep learning to detect plant diseases from leaf images. It features a modern HTML frontend and a Python Flask backend powered by a Keras-trained model. Upload a plant leaf image and get instant predictions about the plant and its health status!

## Features
- Upload a plant leaf image and get a prediction (plant type and health status)
- Beautiful, responsive frontend
- Python Flask backend with TensorFlow/Keras model
- Logging of predictions for analysis
- Easy to run locally

## Project Structure
```
├── backend/
│   └── api_backend.py         # Flask backend API
├── frontend/
│   └── index.html            # HTML/JS/CSS frontend
├── models/
│   └── plant_disease_model.keras  # Trained Keras model
│   └── best_plant_disease_model.h5
├── prediction_log.txt        # Log of predictions
├── Sample Images/            # Example images for testing
├── LeafLine_Plant_Disease_Classification_Model.ipynb  # Model training notebook
```

## Setup & Usage

1. **Clone the repository**

2. **Install dependencies**
   ```powershell
   cd backend
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```
   *(A `requirements.txt` is provided in the `backend/` folder with Flask, flask-cors, tensorflow, pillow, numpy)*

3. **Run the backend**
   ```powershell
   python api_backend.py
   ```
   This will start the Flask server and open the frontend in your browser.

4. **Use the app**
   - Go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) *or your localhost domain if changed*
   - Upload a plant leaf image and view the prediction.

## Notes
- The `venv/` directory (Python virtual environment) should **not** be included in the repository. Add it to `.gitignore`.
- The model file (`plant_disease_model.keras`) is required for predictions. You can retrain or update it using the provided notebook.
- For deployment to cloud or production, further configuration (e.g., Docker, HTTPS, production WSGI server) is recommended.

## License
MIT License

---

*Made with ❤️ using Python, Flask, and TensorFlow.*
