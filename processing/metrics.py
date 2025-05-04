
import numpy as np
from sklearn.metrics import ndcg_score

# funciones de obtencion de metricas
def precision_at_k(obtenidos, esperados, k=10):
    relevantes = set(esperados)
    return len([doc for doc in obtenidos[:k] if doc in relevantes]) / k

def recall_at_k(obtenidos, esperados, k=10):
    relevantes = set(esperados)
    return len([doc for doc in obtenidos[:k] if doc in relevantes]) / len(relevantes)

def f1_at_k(obtenidos, esperados, k=10):
    p_at_k = precision_at_k(obtenidos, esperados, k)
    r_at_k = recall_at_k(obtenidos, esperados, k)
    return 2 * (p_at_k * r_at_k) / (p_at_k + r_at_k) if (p_at_k + r_at_k) > 0 else 0

def success_at_k(obtenidos, esperados, k=10):
    return 1 if any(doc in esperados for doc in obtenidos[:k]) else 0

def mean_average_precision(obtenidos, esperados, k=10):
    relevantes = set(esperados)
    precisions = [precision_at_k(obtenidos, esperados, i + 1) for i in range(k) if obtenidos[i] in relevantes]
    return np.mean(precisions) if precisions else 0

def mean_reciprocal_rank(obtenidos, esperados):
    for i, doc in enumerate(obtenidos):
        if doc in esperados:
            return 1 / (i + 1)
    return 0

def ndcg_at_k(obtenidos, esperados, k=10):
    relevancias = [1 if doc in esperados else 0 for doc in obtenidos[:k]]
    ideal_relevancias = sorted(relevancias, reverse=True)
    return ndcg_score([ideal_relevancias], [relevancias])