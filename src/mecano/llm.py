"""
Appel du LLM (Google Gemini via API REST rapide avec gestion automatique du rate-limit 429).
"""
import re
import time
import requests
from mecano.config import load_config, get_api_key
from mecano.prompts import SYSTEM_PROMPT, MULTIQUERY_SYSTEM_PROMPT, build_multiquery_prompt


def call_gemini_api(
    user_prompt: str,
    system_instruction: str = None,
    model_name: str = "gemini-flash-lite-latest",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 25,
    max_retries: int = 3
) -> str:
    """Appel direct à l'API Gemini via REST avec retry automatique sur code 429."""
    key = get_api_key()
    if not key:
        raise ValueError(
            "Clé GEMINI_API_KEY introuvable. Veuillez la définir dans votre fichier .env "
            "ou variable d'environnement."
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
        }
    }
    if system_instruction:
        payload["system_instruction"] = {
            "parts": [{"text": system_instruction}]
        }

    for attempt in range(max_retries + 1):
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return (parts[0].get("text") or "").strip()
            return ""
        if response.status_code == 429 and attempt < max_retries:
            time.sleep(8 * (attempt + 1))
            continue
        err_msg = response.text
        try:
            err_json = response.json()
            err_msg = err_json.get("error", {}).get("message", err_msg)
        except Exception:
            pass
        raise RuntimeError(f"Erreur API Gemini ({response.status_code}) : {err_msg}")
    return ""


def generate_response(
    user_prompt: str,
    system_prompt: str = None,
    config: dict = None
) -> str:
    """Génère la réponse finale du chatbot avec Gemini."""
    cfg_data = config or load_config()
    llm_cfg = cfg_data.get("llm", {})
    model_name = llm_cfg.get("model_name", "gemini-flash-lite-latest")
    temperature = float(llm_cfg.get("temperature", 0.0))
    max_tokens = int(llm_cfg.get("max_output_tokens", 1024))

    sys_instruction = system_prompt if system_prompt is not None else SYSTEM_PROMPT

    try:
        return call_gemini_api(
            user_prompt=user_prompt,
            system_instruction=sys_instruction,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=25
        )
    except Exception as e:
        return f"Erreur lors de la génération (API Gemini) : {e}"


def generate_reformulations(
    question: str,
    n: int = 3,
    config: dict = None
) -> list[str]:
    """Génère n reformulations de recherche pour la question de l'utilisateur."""
    cfg_data = config or load_config()
    llm_cfg = cfg_data.get("llm", {})
    model_name = llm_cfg.get("model_name", "gemini-flash-lite-latest")

    try:
        user_prompt = build_multiquery_prompt(question)
        raw_text = call_gemini_api(
            user_prompt=user_prompt,
            system_instruction=MULTIQUERY_SYSTEM_PROMPT,
            model_name=model_name,
            temperature=0.0,
            max_tokens=256,
            timeout=15
        )
        lines = []
        for line in raw_text.splitlines():
            l = line.strip()
            l = re.sub(r"^[-*•\d.)\s]+", "", l).strip()
            if l and len(l) > 3 and l not in lines:
                lines.append(l)
        return lines[:n]
    except Exception as e:
        print(f"[Avertissement] Impossible de générer les reformulations : {e}")
        return []
