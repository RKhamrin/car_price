from fastapi import FastAPI
from fastapi import File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pandas as pd 
import numpy as np
import re
import io
import pickle
import uvicorn
import json

app = FastAPI()

num_cols = ['year', 'km_driven', 'mileage',
'engine', 'max_power', 'torque', 'max_torque_rpm']

ohe_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']

fillna_dict = {'year': 2014.0, 'km_driven': 70000.0, 'mileage': 19.369999999999997,
'engine': 1248.0, 'max_power': 81.86, 'torque': 150.0, 'max_torque_rpm': 150.0}

# all_cols = ['Diesel', 'LPG', 'Petrol', 'Individual', 'Trustmark Dealer', 'Manual', 'Fourth & Above Owner',
# 'Second Owner', 'Test Drive Car', 'Third Owner', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun', 'Fiat',
# 'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 'Lexus', 'MG', 'Mahindra',
# 'Maruti', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Peugeot', 'Renault', 'Skoda', 'Tata', 'Toyota',
# 'Volkswagen', 'Volvo']

with open("best_model.pkl", 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open("ohe_cat.pkl", 'rb') as ohe_file:
    loaded_ohe = pickle.load(ohe_file)

# with open("scaler_num.pkl", 'rb') as scaler_file:
#     loaded_scaler = pickle.load(scaler_file)

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> dict:

    item = item.model_dump()
    data = pd.DataFrame(item.values(), index=item.keys()).T
    data['engine'] = data['engine'].dropna().astype(int)
    data['seats'] = data['seats'].dropna().astype(int)
    data['seats'] = data['seats'].fillna(np.median(data['seats']))

    data['mileage'] = pd.to_numeric(data['mileage'].apply(
        lambda x: x.split()[0] if isinstance(x, str) else x
        ))
    data['engine'] = pd.to_numeric(data['engine'].apply(
        lambda x: x.split()[0] if isinstance(x, str) else x
        ))
    data['max_power'] = pd.to_numeric(data['max_power'].apply(
        lambda x: np.nan if x == ' bhp' else x.split()[0] if isinstance(x, str) else x
        ))
    data['torque'] = data['torque'].apply(
        lambda x: float(re.findall(r"\d+[\.,]?\d+", str(x).replace(',', '.'))[0])\
        if len(re.findall(r"\d+[\.,]?\d+", str(x).replace(',', '.'))) != 0 else np.nan
        )
    data['max_torque_rpm'] = data['torque'].apply(
        lambda x: float(re.findall(r"\d+[\.,]?\d+", str(x).replace(',', '.'))[-1])\
        if len(re.findall(r"\d+[\.,]?\d+", str(x).replace(',', '.'))) != 0 else np.nan
        )

    data = data.fillna(fillna_dict)
    data['brand'] = data['name'].apply(lambda x: x.split()[0])
    data = data[num_cols+ohe_cols+['seats']]
    # data[num_cols+['seats']] = loaded_scaler.transform(data[num_cols+['seats']])

    ohe_cols_data = loaded_ohe.transform(data[ohe_cols]).toarray()
    ohe_col_names = loaded_ohe.categories_
    all_cols = []
    [all_cols.extend(cols[1:]) for cols in ohe_col_names]
    ohe_cols_data = pd.DataFrame(ohe_cols_data, columns=all_cols)
    data = pd.concat([
        data.drop(columns=ohe_cols).reset_index(),
        ohe_cols_data.reset_index()
    ], axis=1).drop(columns=['index'])
    res = loaded_model.predict(data[list(loaded_model.feature_names_in_)])
    return {'prediction': res.tolist()}

@app.post("/predict_items")
def predict_items(items: UploadFile = File(...)) -> StreamingResponse:
    
    data = pd.read_csv(items.file, index_col=0)
    data['engine'] = data['engine'].dropna().astype(int)
    data['seats'] = data['seats'].dropna().astype(int)
    data['seats'] = data['seats'].fillna(np.median(data['seats']))

    data['mileage'] = pd.to_numeric(data['mileage'].apply(
        lambda x: x.split()[0] if isinstance(x, str) else x
        ))
    data['engine'] = pd.to_numeric(data['engine'].apply(
        lambda x: x.split()[0] if isinstance(x, str) else x
        ))
    data['max_power'] = pd.to_numeric(data['max_power'].apply(
        lambda x: np.nan if x == ' bhp' else x.split()[0] if isinstance(x, str) else x
        ))
    data['torque'] = data['torque'].apply(
        lambda x: float(re.findall(r"\d+[\.,]?\d+", str(x).replace(',', '.'))[0])\
        if len(re.findall(r"\d+[\.,]?\d+", str(x).replace(',', '.'))) != 0 else np.nan
        )
    data['max_torque_rpm'] = data['torque'].apply(
        lambda x: float(re.findall(r"\d+[\.,]?\d+", str(x).replace(',', '.'))[-1])\
        if len(re.findall(r"\d+[\.,]?\d+", str(x).replace(',', '.'))) != 0 else np.nan
        )

    data = data.fillna(fillna_dict)
    data['brand'] = data['name'].apply(lambda x: x.split()[0])
    data = data[num_cols+ohe_cols+['seats']]
    # data[num_cols+['seats']] = loaded_scaler.transform(data[num_cols+['seats']])

    ohe_cols_data = loaded_ohe.transform(data[ohe_cols]).toarray()
    ohe_col_names = loaded_ohe.categories_
    all_cols = []
    [all_cols.extend(cols[1:]) for cols in ohe_col_names]
    ohe_cols_data = pd.DataFrame(ohe_cols_data, columns=all_cols)
    data = pd.concat([
        data.drop(columns=ohe_cols).reset_index(),
        ohe_cols_data.reset_index()
    ], axis=1).drop(columns=['index'])

    stream = io.StringIO()
    data['preds'] = loaded_model.predict(data[list(loaded_model.feature_names_in_)])
    data.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                 )
    
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response

if __name__ == '__main__':
    uvicorn.run('predictor:app', port=8000, reload=True) #host='127.0.0.1',