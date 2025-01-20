
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# Load the saved objects
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
target_encoder = joblib.load('target_encoder.pkl')
one_hot_encoder = joblib.load('one_hot_encoder.pkl')
color_counts = joblib.load('color_counts.pkl')
interior_counts = joblib.load('interior_counts.pkl')

# Initialize the FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],  # Allows all headers
)
# Function to calculate derived features
def calculate_derived_features(data):
    # 1. Calculate `Overall` based on `odometer`
    if data['odometer'] <= 7200:
        data['Overall'] = 'Poor'
    elif data['odometer'] <= 12150:
        data['Overall'] = 'Fair'
    elif data['odometer'] <= 17600:
        data['Overall'] = 'Good'
    else:
        data['Overall'] = 'Very Good'

    # 2. Map `make` to `made_in`
    us_made = ['chevrolet', 'ford', 'buick', 'cadillac', 'jeep', 'dodge', 'chrysler', 'ram', 'scion', 'pontiac', 'saturn', 'mercury', 'hummer', 'gmc', 'gmc truck', 'oldsmobile', 'ford truck', 'lincoln', 'plymouth', 'airstream']
    germany_made = ['bmw', 'audi', 'mercedes-benz', 'porsche', 'smart', 'chev truck', 'volkswagen']
    japan_made = ['nissan', 'acura', 'lexus', 'infiniti', 'mitsubishi', 'mazda', 'toyota', 'subaru', 'honda', 'suzuki', 'isuzu', 'mazda tk']
    uk_made = ['mini', 'land rover', 'jaguar']
    italy_made = ['fiat', 'maserati']
    korea_made = ['kia', 'hyundai', 'hyundai tk', 'daewoo']
    swedia_made = ['volvo', 'saab']

    if data['make'] in us_made:
        data['made_in'] = 'US'
    elif data['make'] in germany_made:
        data['made_in'] = 'DEU'
    elif data['make'] in japan_made:
        data['made_in'] = 'JPN'
    elif data['make'] in uk_made:
        data['made_in'] = 'UK'
    elif data['make'] in italy_made:
        data['made_in'] = 'ITA'
    elif data['make'] in korea_made:
        data['made_in'] = 'KOR'
    else:
        data['made_in'] = 'SWE'

    # 3. Assign `top_make`
    top_brands = ['toyota', 'ford', 'chevrolet', 'nissan', 'honda', 'bmw', 'mercedes-benz', 'jeep', 'volkswagen', 'hyundai', 'kia']
    if data['make'] in top_brands:
        data['top_make'] = data['make']
    else:
        data['top_make'] = 'Other'

    return data
@app.post('/predict')
async def predict(data: dict):
    try:
        # Convert user input to a DataFrame
        user_data = pd.DataFrame([data])
        
        # Calculate derived features
        user_data = user_data.apply(calculate_derived_features, axis=1)
        
        # Drop columns not used in the model
        user_data.drop(['make'], axis=1, inplace=True)  # 'make' is transformed into 'made_in' and 'top_make'
        
        # Preprocessing
        # Target Encoding
        target_encoded_cols = ['model', 'trim', 'body', 'state']
        user_data[target_encoded_cols] = target_encoder.transform(user_data[target_encoded_cols])
        
        # Count Encoding
        user_data['color'] = user_data['color'].map(color_counts).fillna(0)
        user_data['interior'] = user_data['interior'].map(interior_counts).fillna(0)
        
        # One-Hot Encoding
        ohe_cols = ['Overall', 'made_in', 'top_make']
        user_data_ohe = one_hot_encoder.transform(user_data[ohe_cols])
        user_data_ohe_df = pd.DataFrame(user_data_ohe, columns=one_hot_encoder.get_feature_names_out(ohe_cols))
        user_data.drop(columns=ohe_cols, inplace=True)
        user_data = pd.concat([user_data, user_data_ohe_df], axis=1)
        
        # Scaling
        numerical_cols = ['year', 'odometer', 'condition']
        user_data[numerical_cols] = scaler.transform(user_data[numerical_cols])
        
        # Reorder columns to match training order
        feature_names = joblib.load('feature_names.pkl')  # Load saved feature names
        user_data = user_data[feature_names]
        user_data = user_data[feature_names]
        prediction = model.predict(user_data)
        result = {"predicted_mmr": prediction[0]}
        print(result)  # Log the response for debugging
        return result
    except Exception as e:
        return {"error": str(e)}