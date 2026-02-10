# Projet complet — Assistant juridique RH (RAG) avec LLM **open-source** (Apache-2.0)

Ce projet implémente un assistant **juridique RH** en français, basé sur une architecture **RAG** (recherche + génération).
Par défaut, il utilise **Ollama** avec un modèle **Apache-2.0** (ex: `mistral`).

> Ici on cible des modèles **Apache-2.0** (ex. Mixtral).

## 1) Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Démarrer Ollama + télécharger un modèle Apache-2.0

### Option A (recommandée pour FR): Mistral 
```bash
ollama pull mistral
```


Puis lance le serveur Ollama (si besoin) :
```bash
ollama serve
```

## 3) Construire l'index (mini-corpus fictif fourni)

```bash
cp config.example.yaml config.yaml
python -m src.ingest --input_dir data/samples/corpus --out_dir data/index
```

## 4) Lancer l'API

```bash
uvicorn src.api:app --reload --port 8000
```

Test:
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"Combien de jours de congés payés par an ?","filters":{"country":"FR"}}'
```

## 5) Évaluation retrieval (Recall@K)

```bash
python -m eval.run_eval --index_dir data/index --dataset data/samples/eval_dataset.jsonl
```

## Structure

- `src/` : ingestion, indexation (FAISS + BM25), retrieval hybride, RAG, client LLM (Ollama), API FastAPI
- `data/samples/` : mini-corpus fictif + petit dataset d'évaluation
- `eval/` : script d'évaluation retrieval
- `config.yaml` : configuration (LLM, chemins, retrieval)

## Remarques
- Embeddings: par défaut `intfloat/multilingual-e5-small` (téléchargé au premier run via `sentence-transformers`).
- Ce dépôt fournit un **mini-corpus fictif** : remplace-le par tes docs internes (attention RGPD/confidentialité).
