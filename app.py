from fastapi import FastAPI, UploadFile, File
import joblib
import pandas as pd
app = FastAPI(docs_url="/", title="Deploy DM BIMaster")

# Carrego meu modelo treinado
model = joblib.load("model/spine_model.pkl")

# teste branch
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to make predictions using the trained model.

    :param file: new data to predict

    :return: predictions from the model
    """
    df = pd.read_csv(file.file)
    predictions = model.predict(df)
    return {"prediction": predictions.tolist()}

@app.post("/predict_proba")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    probs = model.predict_proba(df)
    return {"probabilities": probs.tolist()}

@app.get("/models")
async def models():
    """
    Endpoint to return the list of available models.

    :return: list of models
    """
    return {"models": ["Decision Tree", "Random Forest", "Gradient Boosting"]}