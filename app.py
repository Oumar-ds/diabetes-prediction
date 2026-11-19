"""
API de dépistage du diabète - Projet 2
Auteur : KONATE ZANA Oumar

Lancer en local :
    uvicorn app:app --reload
Puis ouvrir http://127.0.0.1:8000/docs pour tester l'API dans le navigateur.

Fichiers du modèle (exportés depuis le notebook), à placer à côté de app.py :
  - modele_xgboost.json     : le modèle XGBoost (format natif, portable)
  - parametres_modele.json  : ordre des colonnes, médianes d'imputation, seuil
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------
# 1. CHARGEMENT DU MODÈLE (une seule fois, au démarrage)
# ---------------------------------------------------------------
DOSSIER = Path(__file__).parent
CHEMIN_MODELE = DOSSIER / "modele_xgboost.json"
CHEMIN_PARAMETRES = DOSSIER / "parametres_modele.json"

for chemin in (CHEMIN_MODELE, CHEMIN_PARAMETRES):
    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}. Place 'modele_xgboost.json' et "
            "'parametres_modele.json' (exportés depuis le notebook) à côté de app.py."
        )

modele = xgb.Booster()
modele.load_model(str(CHEMIN_MODELE))

with open(CHEMIN_PARAMETRES, encoding="utf-8") as f:
    parametres = json.load(f)

colonnes = parametres["colonnes"]    # Ordre exact des variables attendu par le modèle
medianes = parametres["medianes"]    # Médianes du jeu d'entraînement (pour les valeurs absentes)
seuil = parametres["seuil"]          # Seuil de décision choisi (0.45)

AVERTISSEMENT = (
    "Outil de dépistage à but pédagogique, entraîné sur des données de femmes "
    "Pima (Arizona). Il ne remplace pas un diagnostic médical : "
    "consultez un professionnel de santé."
)

# ---------------------------------------------------------------
# 2. FORMAT DES DONNÉES D'ENTRÉE
# Glucose, IMC et Age sont obligatoires. Les autres mesures sont
# facultatives : si elles sont absentes, on utilise la médiane du jeu
# d'entraînement (comme pour les valeurs manquantes du dataset).
# ---------------------------------------------------------------


class Patiente(BaseModel):
    Glucose: float = Field(..., ge=40, le=300,
                           description="Glucose 2 h après test de charge (mg/dL)")
    IMC: float = Field(..., ge=10, le=80,
                       description="Indice de masse corporelle (kg/m²)")
    Age: int = Field(..., ge=18, le=100, description="Âge en années")
    Grossesses: Optional[int] = Field(None, ge=0, le=20,
                                      description="Nombre de grossesses")
    Pression: Optional[float] = Field(None, ge=20, le=150,
                                      description="Pression diastolique (mm Hg)")
    Epaisseur_Peau: Optional[float] = Field(None, ge=1, le=100,
                                            description="Pli cutané du triceps (mm)")
    Insuline: Optional[float] = Field(None, ge=1, le=1000,
                                      description="Insuline 2 h après le test (µU/mL)")
    Heredite: Optional[float] = Field(None, ge=0, le=3,
                                      description="Score d'hérédité diabétique")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Glucose": 148, "IMC": 33.6, "Age": 50,
                "Grossesses": 6, "Pression": 72, "Epaisseur_Peau": 35,
                "Insuline": None, "Heredite": 0.627,
            }
        }
    }


# ---------------------------------------------------------------
# 3. L'API
# ---------------------------------------------------------------
app = FastAPI(
    title="API de dépistage du diabète",
    description="Prédit le risque de diabète à partir de mesures cliniques "
                "(modèle XGBoost entraîné sur le dataset Pima Indians Diabetes).",
    version="1.1.0",
)


@app.get("/")
def accueil():
    return {
        "projet": "Prédiction du diabète - Pima Indians Diabetes",
        "auteur": "KONATE ZANA Oumar",
        "documentation": "/docs",
        "avertissement": AVERTISSEMENT,
    }


@app.get("/health")
def sante():
    return {"statut": "ok", "seuil_de_decision": seuil}


@app.post("/predict")
def predire(patiente: Patiente):
    donnees = {nom: getattr(patiente, nom) for nom in colonnes}
    manquantes = [nom for nom, valeur in donnees.items() if valeur is None]

    # Valeur absente -> médiane d'entraînement (même règle que pendant l'entraînement)
    valeurs = [medianes[nom] if donnees[nom] is None else donnees[nom]
               for nom in colonnes]
    matrice = xgb.DMatrix(np.array([valeurs], dtype=float))

    score = float(modele.predict(matrice)[0])
    diabete = score >= seuil

    return {
        "diabete_predit": bool(diabete),
        "score_risque": round(score, 3),
        "seuil_de_decision": seuil,
        "valeurs_remplacees_par_la_mediane": manquantes,
        "message": (
            "Risque élevé : un test de confirmation est recommandé."
            if diabete else
            "Risque faible selon le modèle, sans garantie."
        ),
        "avertissement": AVERTISSEMENT,
    }
