# How to Create .venv (virtual environment)

## Choose python

pyenv install 3.12.13
pyenv local 3.12.13
or pyenv shell 3.12.13. (for shell terminal)

python -m venv .venv

source .venv/bin/activate

source .venv/bin/activate.fish (for fish terminal)

now you would see you are in the .venv (virtual environment)

then you can try 

python --version  # to check the version of python



pip freeze > requirements.txt     # to get the installed libraries into requirements.txt



curl -X POST http://localhost:5001/california-predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.02381,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'

