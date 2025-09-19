# Chef in My Pocket – Agentic MVP

This is a proof-of-concept AI agent that generates a multi-day meal plan based on a dietary goal (e.g. low-carb, vegetarian, high-protein).  
The agent interacts with a custom MCP server to:

1. Find suitable recipes from a dataset.
2. Build and maintain a shopping list of ingredients.

---

## 🚀 Quick Start

### 0. ENV

Setup openai api key in .env

### 1. Install dependencies

```bash
poetry install
```

### 2. Start the MCP server

```bash
poetry run uvicorn mcp_server.server:app --reload --port 8000
```

### 3. Run the agent

```bash
poetry run python main.py
```

### 4. Example interaction (rather focus on the already tagged recipes to avoid waiting)

```
👨‍🍳 Welcome to Chef in My Pocket! What’s your dietary goal? (low-carb / vegetarian / high-protein), and what do you like to eat? (chicken/beef/vegetables)
Type 'exit' to quit.

You: Hey, id like to get 5 day high-protein plan with chicken and beef
👨‍🍳 How many meals per day? (default 4): 5


```

### 5. Replace a meal

```
User: Replace Day 2
Agent: 🔄 Replacing Day 5..."
```

---

# 🏗️ Architecture Diagram

```plaintext
+-------------------+
|       User        |
+-------------------+
         |
         v
+-------------------+
|      Agent        |  (LangGraph)
| - Parse request   |
| - Draft plan      |
| - Confirm plan    |
| - Execute tools   |
| - JSON memory     |
| - Replace days    |
+-------------------+
   |           |
   v           v
+-------------------+        +------------------+
|   MCP Server      |<-----> | Dataset (CSV)    |
| - find_recipes    |        | + tags (LLM)     |
| - shopping_list   |        | + embeddings     |
| - tag_missing     |        | (Chroma DB)      |
+-------------------+        +------------------+
```

---

# 🧩 Components

### **Agent (LangGraph)**

- Orchestrates meal planning in two phases:

  1. **Planning:**

     - Parse user request into structured state (`goal`, `days`, `meals/day`, `keywords`).
     - Ask LLM to create a draft plan using candidate recipes.
     - Confirm with user before executing.

  2. **Execution:**

     - Calls MCP tools to fetch recipes and build shopping list.
     - Ensures every meal slot is filled (no missing meals).

- **Conversational Memory (JSON):**

  - Stores last plan, shopping list, dietary goal, and preferences.
  - Enables refinements like _“replace day 2”_ or _“add more beef recipes”_.

---

### **MCP Server (FastAPI)**

- `/find_recipes` → **hybrid search**:

  - Filter by dietary goal.
  - If recipes missing this tag → classify all with LLM.
  - Semantic keyword search via **Chroma embeddings**.
  - Combine results into ranked candidate list.

- `/shopping_list` → JSON-based persistence for ingredients.

  - `add` → add new items (deduplicated).
  - `clear` → reset list.
  - `get` → retrieve full list.

- `/tag_missing` → ensure all recipes in dataset are tagged with dietary categories.

---

### **Dataset**

- **Recipes CSV** → raw recipes with title, author, steps, and ingredients.
- **Recipes_tagged.csv** → same data enriched with dietary tags (low-carb, high-protein, vegetarian, etc.).
- **ChromaDB** → persistent embeddings of recipes for semantic keyword search.

---

# 📝 Design Notes

### Why LangGraph?

- Provides **clear separation of planning and execution**.
- Supports structured reasoning and multi-turn corrections.
- Memory buffer makes it easy to refine (“replace Day 2 dinner”).

### Why Chroma + embeddings?

- **Semantic matching** lets users search flexibly (“chicken”, “beef”, “spicy”) without exact keyword matches.
- Combined with dietary tags → stronger filtering than rules alone.
- Persistent Chroma collection avoids re-embedding on every run

### LLM Tagging

- Only triggered when a requested dietary goal is missing from current dataset.
- Tags are cached into `recipes_tagged.csv` to avoid repeated costs.

### Why JSON Memory instead of LangChain’s built-in memory?

- **Persistence**: JSON (`memory.json`) survives process restarts.
- **Control**: We decide what to store (goal, days, meals, plan).
- **Lightweight**: No extra LangChain state management required.
  """

---

# 🚀 Future Improvements

0. **Docker**
1. **Better search and classification** → Now it takes too long using LLM always for missing tags, it should rather add tagged data to chroma, and just look in chroma, but had no time to polish it.
1. **Nutrition-aware planning** → balance macros/calories daily.
1. **UI for planning** → let users approve/modify GPT’s draft plan before execution.
1. **User profiles** → dislikes, allergies, preferred cuisines.
1. **Advanced shopping list** → normalize quantities, export to grocery app/checkout.
1. **Recipe variety scoring** → ensure less repetition across multi-day plans.
1. **Persistent Memory DB** → Store meal plans per user for recall across sessions with some proper DB.

---
