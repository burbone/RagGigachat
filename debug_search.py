from dotenv import load_dotenv
load_dotenv()

from src.embeddings.embedder import LocalEmbedder
from src.retrieval.vector_store import VectorStore

embedder = LocalEmbedder()
vs = VectorStore(embedder)

# 1. Смотрим что вообще есть на стр.8
print("=== ВСЕ ЧАНКИ СО СТР.8 ===")
data = vs._collection.get(where={"page": 8}, include=["documents", "metadatas"])
docs = data.get("documents", [])
metas = data.get("metadatas", [])
print(f"Найдено чанков на стр.8: {len(docs)}")
for i, (doc, meta) in enumerate(zip(docs, metas)):
    print(f"\n-- чанк {i+1} | table={meta.get('is_table')} --")
    print(doc[:400])

# 2. Поиск по разным запросам
print("\n\n=== ПОИСК ===")
for query in [
    "обслуживание СберКарта первый год без комиссии",
    "ТП-231 обслуживание карты",
    "МИР СберКарта тариф стоимость",
    "1.1.1 основной карты без комиссии",
]:
    print(f"\nЗАПРОС: {query}")
    results = vs.search_raw_child(query, top_k=3)
    for r in results:
        print(f"  стр.{r['page']} score={r['score']:.3f} table={r['is_table']}")
        print(f"  {r['child_text'][:150]}")

# 3. Статистика
print(f"\n=== ВСЕГО ЧАНКОВ: {vs._collection.count()} ===")
all_data = vs._collection.get(include=["metadatas"])
pages = sorted(set(m.get("page") for m in all_data["metadatas"]))
print(f"Страницы в индексе: {pages}")