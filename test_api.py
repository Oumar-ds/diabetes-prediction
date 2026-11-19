"""Tests de l'API. Lancer avec : pytest -v"""

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

PATIENTE_COMPLETE = {
    "Glucose": 148, "IMC": 33.6, "Age": 50, "Grossesses": 6,
    "Pression": 72, "Epaisseur_Peau": 35, "Insuline": 100, "Heredite": 0.627,
}


def test_sante():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["statut"] == "ok"


def test_prediction_complete():
    r = client.post("/predict", json=PATIENTE_COMPLETE)
    assert r.status_code == 200
    corps = r.json()
    assert isinstance(corps["diabete_predit"], bool)
    assert 0 <= corps["score_risque"] <= 1
    assert corps["valeurs_remplacees_par_la_mediane"] == []


def test_prediction_avec_champs_obligatoires_seulement():
    r = client.post("/predict", json={"Glucose": 100, "IMC": 25, "Age": 30})
    assert r.status_code == 200
    manquantes = r.json()["valeurs_remplacees_par_la_mediane"]
    assert set(manquantes) == {"Grossesses", "Pression", "Epaisseur_Peau",
                               "Insuline", "Heredite"}


def test_glucose_eleve_donne_un_score_plus_haut_que_glucose_bas():
    bas = client.post("/predict", json={"Glucose": 80, "IMC": 25, "Age": 25}).json()
    haut = client.post("/predict", json={"Glucose": 190, "IMC": 38, "Age": 55}).json()
    assert haut["score_risque"] > bas["score_risque"]


def test_champ_obligatoire_manquant():
    r = client.post("/predict", json={"IMC": 25, "Age": 30})
    assert r.status_code == 422


def test_valeur_impossible_refusee():
    # Un glucose de 0 est une valeur manquante déguisée : l'API la refuse
    r = client.post("/predict", json={"Glucose": 0, "IMC": 25, "Age": 30})
    assert r.status_code == 422
