# Projet — Assistant juridique RH (RAG) avec LLM **open-source** (Apache-2.0)

Assistant **juridique RH** en français basé sur une architecture **RAG** (Retrieval-Augmented Generation) :  
✅ **Recherche** (BM25 + embeddings/FAISS) → ✅ **Sélection de passages** → ✅ **Réponse LLM** avec **citations**.

- Backends LLM supportés :
  - **Ollama** (local) — recommandé pour démarrer
  - **vLLM** exposant une API compatible **OpenAI** (serveur local ou remote)

> ⚠️ Ce projet ne remplace pas un juriste/avocat : il fournit une aide “informatique” basée sur un corpus documentaire.

---

## 1) Prérequis

- Python 3.10+ (recommandé : 3.11)
- (Optionnel) **Ollama** installé si vous utilisez le backend Ollama
- RAM/CPU selon votre modèle (un `mistral`/`mistral:instruct` tourne bien sur une machine standard ; Mixtral/LLama plus gros demandent plus)

---

## 2) Installation (Windows / macOS / Linux)

### Créer un venv + installer les dépendances

#### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Si vous avez une erreur `ModuleNotFoundError: No module named 'requests'` :

```bash
pip install requests
```

(mais idéalement `requests` doit être dans `requirements.txt`).

---

## 3) Configuration

Copiez l’exemple :

```bash
cp config.example.yaml config.yaml
```

### Exemple de configuration (idée générale)

* Backend LLM : `ollama` ou `vllm_openai`
* Index : `data/index`
* Embeddings : `intfloat/multilingual-e5-small` (par défaut)

> **Astuce** : gardez des noms de documents stables (Doc_ID / source / URL) pour des citations propres.

---

## 4) Corpus & ingestion

### Mon corpus est en `.txt` mais c’est du Markdown : problème ?

Non, **pas forcément** : du Markdown reste du texte.
Ce qui compte :

* que l’ingestion lise bien les `.txt` (ou `.md`)
* que le chunking ne “casse” pas trop les sections (titres, listes)
* que la metadata (Doc_ID/source/URL/date) reste proche des chunks

**Conseils pratiques**

* Si votre ingestion filtre par extension, ajoutez `.txt` et `.md`.
* Gardez un en-tête clair par doc (Doc_ID / Pays / URL / date).
* Évitez un document géant unique : préférez plusieurs docs “thématiques”.

### Construire l’index

```bash
python -m src.ingest --input_dir data/samples/corpus --out_dir data/index
```

---

## 5) Lancer l’API (FastAPI)

```bash
uvicorn src.api:app --reload --port 8000
```

### Healthcheck

* `GET /health` → `{ "status": "ok" }`

### Exemple de requête

#### macOS / Linux (curl)

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"Combien de jours de congés payés par mois ?","filters":{"country":"FR"}}'
```

#### Windows (PowerShell)

```powershell
Invoke-RestMethod -Method Post "http://localhost:8000/ask" -ContentType "application/json" -Body '{
  "question": "Combien de jours de congés payés par mois ?",
  "filters": {"country":"FR"}
}'
```

---

## 6) Évaluation

Ce dépôt inclut deux types d’évaluations :

1. **Évaluation retrieval** (Recall@K) — vérifie si les bons documents remontent
2. **Évaluation “full”** (comportement RAG) — refus / couverture / citations / latence

> ⚠️ Le script “full” ne mesure pas la justesse juridique, mais la robustesse + présence de citations.

### 6.1) Évaluation retrieval (Recall@K)

```bash
python -m eval.run_eval --index_dir data/index --dataset data/samples/eval_dataset.jsonl
```

### 6.2) Évaluation complète (RAG full)

Fichier dataset :
`data/samples/35_questions_mix_rag_mistral.jsonl`

#### Windows (PowerShell) — recommandé en **une seule ligne**

```powershell
python eval\eval_full.py --api_base http://localhost:8000 --input_jsonl data\samples\35_questions_mix_rag_mistral.jsonl --timeout 180 --out_csv data\logs\eval_full_35_mix.csv
```

#### Windows (PowerShell) — version multi-ligne (backtick `)

```powershell
python eval\eval_full.py `
  --api_base http://localhost:8000 `
  --input_jsonl data\samples\35_questions_mix_rag_mistral.jsonl `
  --timeout 180 `
  --out_csv data\logs\eval_full_35_mix.csv
```

#### macOS / Linux (bash/zsh)

```bash
python eval/eval_full.py \
  --api_base http://localhost:8000 \
  --input_jsonl data/samples/35_questions_mix_rag_mistral.jsonl \
  --timeout 180 \
  --out_csv data/logs/eval_full_35_mix.csv
```

**Sortie CSV**

* `data/logs/eval_full_35_mix.csv` contient : statut HTTP, latence, détection “refus”, nb citations, preview réponse…

---

## 7) Comment améliorer le score “full” (pratique)

Si vous voulez monter le score global, la clé est souvent :

* **Citation rate** : renvoyer systématiquement au moins 1 source/Doc_ID quand il y a du contexte
* **Réponses non-refus** quand le contexte est suffisant
* **Éviter les patterns de refus** (ex. “je ne peux pas”, “pas assez de contexte”) si vous avez bien des chunks

Bonnes pratiques côté RAG :

* Ajoutez dans la réponse une section finale du style :

  * `Sources: [Doc_ID / URL / titre]`
* Passez au LLM une consigne explicite : *“cite au moins 1 source si des chunks sont fournis”*
* Assurez-vous que votre API renvoie bien `citations: [...]` ou `sources: [...]` (le script les détecte)

---

## 8) Structure du dépôt

* `src/`

  * ingestion, indexation (**FAISS + BM25**), retrieval hybride, RAG, client LLM, API FastAPI
* `data/`

  * `samples/` : corpus & datasets d’exemple (questions `.jsonl`)
  * `index/` : index généré (FAISS / BM25)
  * `logs/` : exports CSV d’évaluation
* `eval/`

  * scripts d’évaluation

---

## 9) Notes

* Embeddings par défaut : `intfloat/multilingual-e5-small` (téléchargé au premier run via `sentence-transformers`).
* Si votre corpus est “chargé” et hétérogène, pensez à :

  * augmenter `top_k_bm25` / `top_k_dense` / `top_k_final`
  * ajuster `alpha` (équilibre BM25 vs dense)
  * réduire la taille des chunks + overlap raisonnable (meilleures citations, moins de bruit)

