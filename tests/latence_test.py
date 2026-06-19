"""
tests/load_test.py
Tests de latence pour l'API de détection d'anomalies radiologiques.
Lance avec : python tests/load_test.py
"""

import os
import requests
import time
import threading
import io
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
SEUIL_LATENCE = 2.0  # secondes — seuil défini dans main.py


def get_auth_headers() -> dict:
    """Obtient un token JWT et retourne le header Authorization."""
    username = os.getenv("TEST_USERNAME")
    password = os.getenv("TEST_PASSWORD")
    if not username or not password:
        raise RuntimeError("TEST_USERNAME et TEST_PASSWORD doivent être définis dans .env")
    r = requests.post(
        f"{BASE_URL}/token",
        data={"username": username, "password": password},
    )
    r.raise_for_status()
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def make_fake_image(size=(224, 224)):
    """Crée une image factice en mémoire."""
    img = Image.new('RGB', size, color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


# ─── TEST 1 : latence simple (20 requêtes séquentielles) ───────────────────
def test_latence_simple(headers):
    print("\n=== TEST 1 : Latence simple (20 requêtes séquentielles) ===")
    latences = []
    for i in range(20):
        img = make_fake_image()
        r = requests.post(f"{BASE_URL}/predict", files={"image": ("test.png", img, "image/png")}, headers=headers)
        latence = r.elapsed.total_seconds()
        latences.append(latence)
        print(f"  [{i+1:02d}] {r.json()} — {latence:.3f}s")
        time.sleep(1)
    print(f"  → Latence moyenne : {sum(latences)/len(latences):.3f}s")
    print(f"  → Latence max     : {max(latences):.3f}s")


# ─── TEST 2 : latence sous charge (10 requêtes simultanées) ────────────────
def test_latence_charge(headers):
    print("\n=== TEST 2 : Latence sous charge (10 requêtes simultanées) ===")
    resultats = []
    lock = threading.Lock()

    def envoyer_requete(i):
        img = make_fake_image()
        r = requests.post(f"{BASE_URL}/predict", files={"image": ("test.png", img, "image/png")}, headers=headers)
        latence = r.elapsed.total_seconds()
        with lock:
            resultats.append(latence)
            print(f"  [thread-{i:02d}] {r.json()} — {latence:.3f}s")

    threads = [threading.Thread(target=envoyer_requete, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"  → Latence moyenne sous charge : {sum(resultats)/len(resultats):.3f}s")
    print(f"  → Latence max sous charge     : {max(resultats):.3f}s")


# ─── TEST 3 : assertion seuil 2s ───────────────────────────────────────────
def test_seuil_latence(headers):
    print("\n=== TEST 3 : Assertion seuil 2s (10 requêtes) ===")
    echecs = []
    for i in range(10):
        img = make_fake_image()
        r = requests.post(f"{BASE_URL}/predict", files={"image": ("test.png", img, "image/png")}, headers=headers)
        latence = r.elapsed.total_seconds()
        status = "✓" if latence < SEUIL_LATENCE else "✗ DÉPASSEMENT"
        print(f"  [{i+1:02d}] {latence:.3f}s — {status}")
        if latence >= SEUIL_LATENCE:
            echecs.append(latence)
        time.sleep(0.5)

    if echecs:
        print(f"  → ÉCHEC : {len(echecs)} requête(s) ont dépassé {SEUIL_LATENCE}s")
    else:
        print(f"  → SUCCÈS : toutes les requêtes sous {SEUIL_LATENCE}s")


# ─── TEST 4 : latence avec grande image (simulation radio réelle) ───────────
def test_latence_grande_image(headers):
    print("\n=== TEST 4 : Latence grande image 1024x1024 (simulation radio réelle) ===")
    latences_small = []
    latences_large = []

    for i in range(5):
        img_small = make_fake_image(size=(224, 224))
        r = requests.post(f"{BASE_URL}/predict", files={"image": ("small.png", img_small, "image/png")}, headers=headers)
        latences_small.append(r.elapsed.total_seconds())

        img_large = make_fake_image(size=(1024, 1024))
        r = requests.post(f"{BASE_URL}/predict", files={"image": ("large.png", img_large, "image/png")}, headers=headers)
        latences_large.append(r.elapsed.total_seconds())
        time.sleep(0.5)

    print(f"  → Latence moyenne 224x224  : {sum(latences_small)/len(latences_small):.3f}s")
    print(f"  → Latence moyenne 1024x1024: {sum(latences_large)/len(latences_large):.3f}s")
    diff = sum(latences_large)/len(latences_large) - sum(latences_small)/len(latences_small)
    print(f"  → Surcoût preprocessing    : +{diff:.3f}s")


if __name__ == "__main__":
    headers = get_auth_headers()
    test_latence_simple(headers)
    test_latence_charge(headers)
    test_seuil_latence(headers)
    test_latence_grande_image(headers)
    print("\n=== Tous les tests terminés — vérifiez Grafana pour les courbes ===")