"""Utilidades para parsear JSON de salidas de modelos locales."""

import json
import re


def clean_json_output(text: str) -> str:
    """Limpia posible markdown o espacios extras alrededor del JSON."""
    text = text.strip()
    # Elimina bloques de código markdown si existen
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_output(text: str) -> dict:
    """Parsea un string a dict, limpiando primero el formato."""
    cleaned = clean_json_output(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Último intento: extraer el primer objeto JSON válido
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"No se pudo parsear JSON: {exc}\nTexto: {cleaned}") from exc
