import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': "https://escolarai-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "656229":
        {
            "name": "Marcelo Leite",
            "class": "Manha 7 A",
            "starting_year": 2017,
            "total_presence": 7,
            "last_presence_time": "2024-07-02 00:54:34"
        },
    "852741":
        {
            "name": "Emly Blunt",
            "class": "Manha 7 A",
            "starting_year": 2021,
            "total_presence": 12,
            "last_presence_time": "2024-07-02 00:54:34"
        },
    "963852":
        {
            "name": "Elon Musk",
            "class": "Tarde 8 D",
            "starting_year": 2020,
            "total_presence": 7,
            "last_presence_time": "2024-07-02 00:54:34"
        }
}

for key, value in data.items():
    ref.child(key).set(value)