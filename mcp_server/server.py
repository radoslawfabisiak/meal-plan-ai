# mcp_server/server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
import json
import chromadb

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Chef MCP Server")


# -----------------------------
# Schemas
# -----------------------------
class RecipeQuery(BaseModel):
    dietary_goal: Optional[str] = None
    keywords: Optional[List[str]] = None
    limit: int = 3


class RecipeResult(BaseModel):
    title: str
    ingredients: List[str]


class ShoppingListAction(BaseModel):
    action: str  # "add", "get", "clear"
    items: Optional[List[str]] = None


class ShoppingListResponse(BaseModel):
    items: List[str]


# -----------------------------
# Config
# -----------------------------
RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recipes.csv")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recipes_tagged.csv")
SHOPPING_LIST_FILE = os.path.join(os.path.dirname(__file__), "shopping_list.json")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")

EMBED_MODEL = "text-embedding-3-small"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Ensure recipes_tagged.csv exists
# -----------------------------
if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0:
    try:
        recipes_df = pd.read_csv(DATA_PATH)
    except Exception:
        print("⚠️ recipes_tagged.csv corrupted or empty, recreating from recipes.csv...")
        raw_df = pd.read_csv(RAW_PATH)
        raw_df["tags"] = ["[]" for _ in range(len(raw_df))]
        raw_df.to_csv(DATA_PATH, index=False)
        recipes_df = raw_df
else:
    raw_df = pd.read_csv(RAW_PATH)
    raw_df["tags"] = ["[]" for _ in range(len(raw_df))]
    raw_df.to_csv(DATA_PATH, index=False)
    recipes_df = raw_df


# -----------------------------
# Chroma client
# -----------------------------
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection("recipes")


# -----------------------------
# Shopping list persistence
# -----------------------------
def load_shopping_list() -> List[str]:
    if os.path.exists(SHOPPING_LIST_FILE):
        with open(SHOPPING_LIST_FILE, "r") as f:
            return json.load(f)
    return []


def save_shopping_list(items: List[str]):
    with open(SHOPPING_LIST_FILE, "w") as f:
        json.dump(items, f)


shopping_list: List[str] = load_shopping_list()


# -----------------------------
# Helper: normalize tags into a list of strings
# -----------------------------
def normalize_tags(val):
    """Convert tags from various formats to a clean list of lowercase strings."""
    if val is None or pd.isna(val):
        return []

    if isinstance(val, list):
        return [str(t).lower().strip() for t in val]

    if isinstance(val, str):
        # Handle empty strings or whitespace
        val = val.strip()
        if not val or val == "[]":
            return []

        try:
            # Try to parse as JSON first
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(t).lower().strip() for t in parsed]
            else:
                return [str(parsed).lower().strip()]
        except (json.JSONDecodeError, ValueError):
            # If not JSON, treat as single tag
            return [val.lower().strip()]

    return []


# -----------------------------
# Helper: classify dynamically
# -----------------------------
def classify_with_llm(
    title: str, ingredients: str, requested_goal: Optional[str] = None
) -> List[str]:
    """
    Dynamic classification. Supports default categories (vegetarian, low-carb, high-protein)
    and user-requested ones like "high-carb", "gluten-free", etc.
    """
    base_categories = ["vegetarian", "low-carb", "high-protein"]
    if requested_goal and requested_goal not in base_categories:
        base_categories.append(requested_goal)

    categories_text = ", ".join(base_categories)

    prompt = f"""
    You are a nutrition assistant. Classify this recipe into categories:
    {categories_text}

    Recipe: {title}
    Ingredients: {ingredients}

    Respond ONLY with a JSON list of categories, e.g.: ["vegetarian","low-carb"]
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return []


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/find_recipes", response_model=List[RecipeResult])
def find_recipes(query: RecipeQuery):
    """
    Hybrid search:
    - Filter by dietary tags (if present).
    - If no recipes have the requested tag, classify those recipes dynamically with LLM.
    - Semantic keyword search via Chroma.
    """
    global recipes_df
    if recipes_df.empty:
        return []

    df = recipes_df.copy()

    # --- Step 1: dietary goal filter ---
    if query.dietary_goal:
        goal = query.dietary_goal.lower().strip()
        print(f"🔎 Checking if recipes are tagged for '{goal}'...")

        # Check which recipes need classification
        needs_classification = []
        recipes_with_goal = []

        for idx, row in df.iterrows():
            tags = normalize_tags(row.get("tags", []))
            print(
                f"Recipe {idx}: {row.get('name', 'Unknown')} has tags: {tags}"
            )  # Debug line

            if goal in tags:
                recipes_with_goal.append(idx)
                print(f"✅ Recipe {idx} already has '{goal}' tag")  # Debug line
            elif not tags:  # only classify if no tags at all
                needs_classification.append((idx, tags, row))

        print(f"📊 Found {len(recipes_with_goal)} recipes already tagged with '{goal}'")
        print(f"📊 Found {len(needs_classification)} recipes needing classification")

        # Only classify if we have recipes that need it
        if not recipes_with_goal and needs_classification:
            print(f"⚡ Classifying {len(needs_classification)} recipes...")
            changes_made = False

            for idx, existing_tags, row in needs_classification:
                new_tags = classify_with_llm(
                    row.get("name", ""), row.get("ingredients", ""), goal
                )
                if new_tags:
                    # Merge existing tags with new ones
                    combined_tags = list(
                        set(existing_tags + [t.lower().strip() for t in new_tags])
                    )
                    df.at[idx, "tags"] = json.dumps(combined_tags)
                    changes_made = True
                    print(f"🏷️ Recipe {idx} updated with tags: {combined_tags}")

            # Save changes if any were made
            if changes_made:
                df.to_csv(DATA_PATH, index=False)
                recipes_df = df  # Update global DataFrame
                print("✅ Classification complete and saved.")
            else:
                print("ℹ️ No new classifications were made.")

        # Now filter recipes that have the goal
        df_filtered = df.copy()
        filtered_indices = []

        for idx, row in df_filtered.iterrows():
            tags = normalize_tags(row.get("tags", []))
            if goal in tags:
                filtered_indices.append(idx)

        df = df.iloc[filtered_indices] if filtered_indices else pd.DataFrame()
        print(f"🍽️ {len(df)} recipes match dietary goal '{goal}' after processing.")

    # --- Step 2: keyword search via embeddings ---
    if query.keywords:
        embeddings = [
            client.embeddings.create(model=EMBED_MODEL, input=kw).data[0].embedding
            for kw in query.keywords
        ]
        results = collection.query(query_embeddings=embeddings, n_results=10)
        ids = set(results["ids"][0]) if results["ids"] else set()
        if not df.empty:
            df = df[df["id"].astype(str).isin(ids)]
        else:
            df = recipes_df[recipes_df["id"].astype(str).isin(ids)]

    # --- Step 3: build response ---
    recipes = []
    for _, row in df.head(query.limit).iterrows():
        ingredients = []
        if "ingredients" in row and not pd.isna(row["ingredients"]):
            ingredients = [i.strip() for i in str(row["ingredients"]).split(",")]

        recipes.append(
            {"title": row.get("name", "Unknown Recipe"), "ingredients": ingredients}
        )

    return recipes


@app.post("/shopping_list", response_model=ShoppingListResponse)
def shopping_list_manager(action: ShoppingListAction):
    global shopping_list

    if action.action == "add" and action.items:
        shopping_list.extend(action.items)
        shopping_list = list(set(shopping_list))
        save_shopping_list(shopping_list)
    elif action.action == "clear":
        shopping_list = []
        save_shopping_list(shopping_list)

    return {"items": shopping_list}


@app.post("/tag_missing")
def tag_missing():
    """
    Classify all recipes without tags using LLM (default 3 categories).
    """
    global recipes_df
    changed = False

    if "tags" not in recipes_df.columns:
        recipes_df["tags"] = ["[]" for _ in range(len(recipes_df))]

    for idx, row in recipes_df.iterrows():
        tags = normalize_tags(row.get("tags", []))

        if not tags:  # Only classify if no tags exist
            new_tags = classify_with_llm(
                row.get("name", ""), row.get("ingredients", "")
            )
            if new_tags:
                recipes_df.at[idx, "tags"] = json.dumps(new_tags)
                changed = True

    if changed:
        recipes_df.to_csv(DATA_PATH, index=False)
        print("✅ Recipes tagged and saved.", flush=True)
    else:
        print("ℹ️ No untagged recipes found.", flush=True)

    return {"message": "Tagging complete", "changed": changed}
