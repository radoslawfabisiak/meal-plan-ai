# scripts/embed_recipes.py
import csv
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/recipes.csv"
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "recipes"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Clear & recreate collection
try:
    chroma_client.delete_collection(COLLECTION_NAME)
except:
    pass
collection = chroma_client.create_collection(COLLECTION_NAME)


def embed_text(text: str):
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding


def main():
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row["id"]
            name = row["name"]
            ingredients = row["ingredients"]
            steps = row["steps"]

            # Build searchable text
            text = f"{name}. Ingredients: {ingredients}. Steps: {steps}"

            emb = embed_text(text)

            collection.add(
                ids=[doc_id],
                embeddings=[emb],
                documents=[text],
                metadatas=[{"name": name}],
            )
            print(f"✅ Embedded: {name}")


if __name__ == "__main__":
    main()
