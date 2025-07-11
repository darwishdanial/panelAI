from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model
model = joblib.load('panel_model.joblib')

# Define the expected input format
class ProjectInput(BaseModel):
    project_area: int
    project_type: int

@app.post("/predict-panel")
def predict_panel(input: ProjectInput):
    # Prepare the input for prediction
    # X_input = np.array([[input.project_area, input.project_type]])

    X_input = pd.DataFrame([{
        "type_encoded": input.project_type,
        "area_encoded": input.project_area
    }])
    
    # Get prediction probabilities
    probabilities = model.predict_proba(X_input)[0]
    class_indices = model.classes_

    # Return as dictionary
    result = {str(class_id): round(prob, 4) for class_id, prob in zip(class_indices, probabilities)}
    return {"predictions": result}