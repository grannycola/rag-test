from pymilvus import connections, Collection

# Подключаемся
connections.connect(alias="default", host="172.21.1.4", port="19530")

# Открываем коллекцию
col = Collection("rag_chunks")

# Сколько всего чанков
print("Всего записей:", col.num_entities)

# Простой выбор первых N строк
res = col.query(
    expr="chunk_idx >= 0",
    output_fields=["doc_path", "chunk_idx", "text"],
    limit=5
)

for r in res:
    print(r)

# Список индексов на коллекции
for idx in col.indexes:
    print("field:", idx.field_name)
    print("index_name:", idx.index_name)            # может быть None/'' по умолчанию
    print("index_type:", idx.params.get("index_type"))
    print("metric_type:", idx.params.get("metric_type"))
    print("params:", idx.params.get("params"))
    print("-"*40)