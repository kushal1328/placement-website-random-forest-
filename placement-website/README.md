# Placement Outlook

A website that predicts whether a student will be **Placed** or **Not placed**.
For every prediction it shows both the **class** and the **probability**.

## Run it on your computer

1. Install Python 3.10 or newer from python.org (tick "Add Python to PATH" on Windows).
2. Open a terminal in this folder.
3. Create a virtual environment:
   - Windows:   `python -m venv venv`  then  `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then  `source venv/bin/activate`
4. Install the libraries:  `pip install -r requirements.txt`
5. Train the model (about a minute):  `python train_model.py`
6. Start the website:  `python app.py`
7. Open  http://127.0.0.1:5000  in your browser.

Stop the server with Ctrl+C.

## Files

| File | Purpose |
|---|---|
| `placement_train.csv` | The dataset |
| `train_model.py` | Cleans data, trains and evaluates the models, saves them to `model/` |
| `app.py` | Flask server: serves the page and the `/api/predict` endpoint |
| `templates/index.html` | The page |
| `static/css/style.css`, `static/js/app.js` | Styling and browser logic |

## The API

`POST /api/predict` with JSON such as:

    {"CGPA": 7.5, "Aptitude_Test_Score": 75, "Backlogs": 0, "Internships": 1, "Projects": 4,
     "Certifications": 2, "Coding_Skills": 6, "Communication_Skills": 5, "Soft_Skills_Rating": 6,
     "Degree": "B.Tech", "Branch": "IT"}

The response contains `prediction.class` (0 or 1), `prediction.label`, and `prediction.probability_placed`.

## Put it online (Render)

1. Push this folder to a new GitHub repository (keep `placement_train.csv` in it).
2. On render.com choose New > Web Service and connect the repository.
3. Language: Python 3. Build command: `pip install -r requirements.txt && python train_model.py`
   Start command: `gunicorn app:app`
4. Create the service. Render gives you a public https://....onrender.com address.
