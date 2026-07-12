```
curl -X POST http://127.0.0.1:5001/california-predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 3.5,
    "HouseAge": 20,
    "AveRooms": 5.0,
    "AveBedrms": 1.2,
    "Population": 1000,
    "AveOccup": 3,
    "Latitude": 35.0,
    "Longitude": -118.0
  }'
```