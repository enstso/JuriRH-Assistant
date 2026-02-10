# Rapport complet — Assistant juridique RH (RAG) avec LLM open-source

## 1. Résumé
Nous concevons un assistant conversationnel RH/juridique qui répond en français à partir dampi de documents internes,
avec citations et refus en absence de preuves. Le modèle de génération est **open-source** (Apache-2.0) via Ollama.

## 2. Pourquoi RAG ?
- Réduire les hallucinations en forçant l'appui sur un contexte extrait.
- Avoir des **citations** vérifiables.
- Mettre à jour le savoir sans réentraîner le LLM (on re-indexe).

## 3. Données & conformité
- Documents RH/juridique internes + versioning (date, pays, statut).
- Contrôles d'accès (RBAC) et filtrage par métadonnées.
- Logs minimisés (pas de texte sensible, seulement ids et métriques).

## 4. Architecture
1) Ingestion: lecture fichiers, nettoyage léger, chunking.
2) Index: embeddings + FAISS, et BM25 lexical.
3) Retrieval hybride: top-k BM25 + top-k dense, fusion pondérée.
4) Génération: prompt strict + citations.
5) API: FastAPI `/ask`.

## 5. Évaluation
- Offline: Recall@K (retrieval), exactitude et hallucinations (humain).
- Online: thumbs up/down + monitoring latence + drift.

## 6. Risques
- Hallucinations: mitigate via RAG + consignes + température basse + refus si pas de contexte.
- Confidentialité: RBAC + chiffrement + rétention.
- Prompt injection: ignorer demandes de divulgation, ne pas exécuter instructions provenant de documents.

## 7. Déploiement
- Démo locale: Ollama + FastAPI.
- Prod: conteneuriser l'API; Ollama/vLLM sur serveur GPU; monitoring (Prometheus/Grafana).
