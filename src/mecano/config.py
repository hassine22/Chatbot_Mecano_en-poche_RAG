"""
Lecture de config.yaml et des variables d'environnement (.env).
"""
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Racines possibles du projet
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def get_project_root() -> Path:
    return ROOT_DIR


def load_config(config_path: str = None) -> dict:
    """Charge config.yaml et les variables de .env."""
    load_dotenv(ROOT_DIR / ".env")
    
    if config_path is None:
        p = ROOT_DIR / "config.yaml"
    else:
        p = Path(config_path)
        if not p.is_absolute():
            p = ROOT_DIR / p
            
    if not p.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {p}")
        
    with open(p, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
        
    return config


def get_api_key(var_name: str = "GEMINI_API_KEY") -> str:
    """Récupère la clé API depuis l'environnement ou .env."""
    load_dotenv(ROOT_DIR / ".env")
    key = os.getenv(var_name, "").strip()
    return key
