# main.py
import os
import sys
import re
import json
import random
import requests
from typing import List, Dict, Any, Set
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

MCP_URL = "http://127.0.0.1:8000"
MEMORY_FILE = "memory.json"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# -----------------------------
# Memory Helpers
# -----------------------------
def save_memory(state: "AgentState | dict"):
    if isinstance(state, dict):
        state = AgentState(**state)  # normalize dict → model

    data = state.model_dump()
    if isinstance(data.get("used_ids"), set):
        data["used_ids"] = list(data["used_ids"])
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_memory() -> "AgentState | None":
    if not os.path.exists(MEMORY_FILE):
        return None
    try:
        with open(MEMORY_FILE, "r") as f:
            content = f.read().strip()
            if not content:  # empty file
                return None
            data = json.loads(content)
    except json.JSONDecodeError:
        print("⚠️ memory.json is invalid or empty, starting fresh.", flush=True)
        return None

    # Normalize used_ids
    if isinstance(data.get("used_ids"), list):
        data["used_ids"] = set(data["used_ids"])

    # 🔧 Normalize plan if it’s stored as dict {"Day 1": [...], ...}
    if isinstance(data.get("plan"), dict):
        normalized_plan = []
        for day, meals in data["plan"].items():
            daily_meals = []
            for idx, meal in enumerate(meals, start=1):
                daily_meals.append({"meal": idx, "title": meal, "ingredients": []})
            normalized_plan.append({"day": day, "meals": daily_meals})
        data["plan"] = normalized_plan

    return AgentState(**data)


# -----------------------------
# Agent State
# -----------------------------
class AgentState(BaseModel):
    goal: str = ""
    days: int = 0
    meals_per_day: int = 0
    keywords: List[str] = []
    plan: List[Dict[str, Any]] = []
    shopping_list: List[str] = []
    used_ids: Set[str] = set()


def replace_day(state: AgentState, day_number: int) -> AgentState:
    if isinstance(state, dict):  # 🔧 fix for dict input
        state = AgentState(**state)
    day_label = f"Day {day_number}"

    print(f"🔄 Replacing {day_label}...", flush=True)

    # Collect candidates
    tag_candidates = recipe_finder(state.goal, None, limit=50)
    keyword_candidates = []
    for kw in state.keywords:
        keyword_candidates.extend(recipe_finder(state.goal, [kw], limit=50))
    all_candidates = {c["title"]: c for c in (tag_candidates + keyword_candidates)}

    recipe_names = list(all_candidates.keys())

    if not recipe_names:
        print(f"⚠️ No recipes found for {day_label}. Cannot replace this day.")
        return state

    # Build only this day’s plan
    prompt = f"""
    You are a meal planner. From this list: {recipe_names}
    Create ONLY {state.meals_per_day} meals for {day_label}.
    Ensure diversity, avoid repeats if possible.
    Return ONLY valid JSON in this format:
    {{"{day_label}": ["Recipe A", "Recipe B", ...]}}
    """

    try:
        resp = llm.invoke(
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        plan_json = json.loads(resp.content)
    except Exception:
        print("⚠️ LLM failed, falling back sequentially", flush=True)
        plan_json = {
            day_label: (
                random.sample(recipe_names, k=state.meals_per_day)
                if len(recipe_names) >= state.meals_per_day
                else [random.choice(recipe_names) for _ in range(state.meals_per_day)]
            )
        }

    # Remove old meals for this day from plan + shopping list
    old_day = next((d for d in state.plan if d["day"] == day_label), None)
    if old_day:
        for meal in old_day["meals"]:
            for ing in meal["ingredients"]:
                try:
                    state.shopping_list.remove(ing)
                except ValueError:
                    pass
        state.plan = [d for d in state.plan if d["day"] != day_label]

    # Add new meals
    daily_meals = []
    for meal_idx, meal_name in enumerate(plan_json[day_label], start=1):
        recipe = all_candidates.get(meal_name)
        if not recipe:
            continue
        print(f"🥘 {day_label}, Meal {meal_idx}: {meal_name}", flush=True)
        print("🛒 Adding ingredients...", flush=True)
        daily_meals.append(
            {
                "meal": meal_idx,
                "title": recipe["title"],
                "ingredients": recipe["ingredients"],
            }
        )
        state.shopping_list = shopping_list_manager("add", recipe["ingredients"])

    state.plan.append({"day": day_label, "meals": daily_meals})
    print(f"✅ {day_label} replaced!\n", flush=True)

    # Keep days in order
    state.plan = sorted(state.plan, key=lambda d: int(d["day"].split()[1]))

    save_memory(state)
    return state


# -----------------------------
# MCP Tool wrappers
# -----------------------------
def recipe_finder(
    goal: str, keywords: List[str] = None, limit: int = 1
) -> List[Dict[str, Any]]:
    payload = {"dietary_goal": goal, "limit": limit}
    if keywords:
        payload["keywords"] = keywords
    print(
        f"🔍 Querying MCP for recipes (goal='{goal}', keywords={keywords})... If data is not tagged properly it can take few minutes with first runs, to prepare our data. You can take a look on the backend logs to see whats going on.",
        flush=True,
    )
    resp = requests.post(f"{MCP_URL}/find_recipes", json=payload)
    try:
        data = resp.json()
        print(f"   → {len(data)} recipes returned.", flush=True)
        return data
    except Exception as e:
        print(f"⚠️ recipe_finder failed: {e}", flush=True)
        return []


def shopping_list_manager(action: str, items: List[str] = None) -> List[str]:
    payload = {"action": action}
    if items:
        payload["items"] = items
    resp = requests.post(f"{MCP_URL}/shopping_list", json=payload)
    return resp.json().get("items", [])


# -----------------------------
# Parse user input with LLM
# -----------------------------
def parse_user_input(text: str) -> AgentState:
    state = AgentState()
    prompt = f"""
    Extract structured meal plan request from user input.
    Return JSON with keys:
    - goal: dietary goal (low-carb, vegetarian, high-protein, high-carb, gluten-free, or "")
    - days: integer (0 if not specified)
    - meals_per_day: integer (0 if not specified)
    - keywords: list of strings
    """

    try:
        resp = llm.invoke([{"role": "user", "content": prompt}])
        parsed = json.loads(resp.content.strip())
        state.goal = parsed.get("goal", "")
        state.days = parsed.get("days", 0)
        state.meals_per_day = parsed.get("meals_per_day", 0)  # 🔧 keep it!
        state.keywords = parsed.get("keywords", [])
    except Exception:
        # Regex fallback
        match_goal = re.search(
            r"(low-carb|vegetarian|high-protein|high-carb|gluten-free)", text, re.I
        )
        if match_goal:
            state.goal = match_goal.group(1).lower()
        match_days = re.search(r"(\d+)\s*day", text, re.I)
        if match_days:
            state.days = int(match_days.group(1))
        match_meals = re.search(r"(\d+)\s*meals?", text, re.I)
        if match_meals:
            state.meals_per_day = int(match_meals.group(1))

    return state


# -----------------------------
# Agent workflow
# -----------------------------
def build_plan(state: AgentState, replace_day: int = None) -> AgentState:
    print("🤔 Thinking about your plan...", flush=True)

    # Step 1: candidate recipes
    print("📥 Getting candidate recipes...", flush=True)
    tag_candidates = recipe_finder(state.goal, state.keywords, limit=50)

    keyword_candidates = []
    for kw in state.keywords:
        keyword_candidates.extend(recipe_finder(state.goal, [kw], limit=50))

    all_candidates = {c["title"]: c for c in (tag_candidates + keyword_candidates)}
    recipe_names = list(all_candidates.keys())

    if not recipe_names:  # 🔧 Guard fix
        print(
            "⚠️ No recipes found for your goal/keywords. Try different keywords or goal."
        )
        return state  # bail out gracefully

    print(f"📊 {len(recipe_names)} candidate recipes collected.", flush=True)

    # Step 2: draft plan
    print("🧮 Preparing draft plan with LLM...", flush=True)
    prompt = f"""
    You are a professional meal planner.
    From this list of recipes: {recipe_names}

    Create a {state.days}-day plan with {state.meals_per_day} meals per day.
    Rules:
    - Use only recipes from the list provided.
    - Ensure diversity (if keywords include chicken and beef, both appear).
    - Avoid excessive repetition.
    - Return ONLY valid JSON.
    """
    try:
        resp = llm.invoke(
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        plan_json = json.loads(resp.content)
    except Exception:
        print("⚠️ LLM failed, falling back sequentially", flush=True)
        plan_json, idx = {}, 0
        for d in range(1, state.days + 1):
            meals = []
            for _ in range(state.meals_per_day):
                meals.append(recipe_names[idx % len(recipe_names)])
                idx += 1
            plan_json[f"Day {d}"] = meals

    # Step 3: execute plan
    print("🛠 Executing plan and building shopping list...\n", flush=True)
    if replace_day:
        state.shopping_list = []  # reset to rebuild
        state.plan = [d for d in state.plan if d["day"] != f"Day {replace_day}"]

    new_plan = [] if replace_day else []

    for d in range(1, state.days + 1):
        if replace_day and d != replace_day:
            continue  # skip untouched days
        day_label = f"Day {d}"
        meals = plan_json.get(day_label, [])

        # normalize
        meals = [m for m in meals if m and isinstance(m, str)]
        while len(meals) < state.meals_per_day:
            meals.append(random.choice(recipe_names))
        if len(meals) > state.meals_per_day:
            meals = meals[: state.meals_per_day]

        daily_meals = []
        for meal_idx, meal_name in enumerate(meals, start=1):
            recipe = all_candidates.get(meal_name)
            if not recipe:
                continue
            print(f"🥘 {day_label}, Meal {meal_idx}: {meal_name}", flush=True)
            print("🛒 Adding ingredients...", flush=True)
            daily_meals.append(
                {
                    "meal": meal_idx,
                    "title": recipe["title"],
                    "ingredients": recipe["ingredients"],
                }
            )
            state.shopping_list = shopping_list_manager("add", recipe["ingredients"])
        new_plan.append({"day": day_label, "meals": daily_meals})
        print(f"✅ {day_label} complete!\n", flush=True)

    # merge replaced/new days
    state.plan.extend(new_plan)
    state.plan = sorted(state.plan, key=lambda d: d["day"])

    # Final shopping list
    print("🛒 Final Shopping List:\n" + ", ".join(state.shopping_list), flush=True)

    save_memory(state)
    return state


# -----------------------------
# LangGraph workflow
# -----------------------------
workflow = StateGraph(AgentState)
workflow.add_node("build_plan", build_plan)
workflow.set_entry_point("build_plan")
workflow.add_edge("build_plan", END)
app = workflow.compile()


# -----------------------------
# CLI Loop
# -----------------------------
def run_cli():
    print(
        "👨‍🍳 Welcome to Chef in My Pocket! "
        "What’s your dietary goal? (low-carb / vegetarian / high-protein), "
        "and what do you like to eat? (chicken/beef/vegetables)"
    )
    print("Type 'exit' to quit, or 'replace day X' to redo a day.\n")

    state = load_memory()
    if state:
        print("✅ Loaded previous session from memory!\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            sys.exit(0)

        # --- Handle replace command ---
        match = re.match(r"replace day (\d+)", user_input.lower())
        if match:
            if not state:
                print(
                    "⚠️ No active plan to replace a day. Start by making a new plan first."
                )
                continue
            day_number = int(match.group(1))
            state = replace_day(state, day_number)
            continue

        # --- Normal flow (new plan) ---
        state = parse_user_input(user_input)

        if not state.goal:
            state.goal = input("👨‍🍳 What’s your dietary goal?: ").strip()
        if not state.days:
            state.days = int(input("👨‍🍳 How many days should I plan for?: ").strip())
        if not state.meals_per_day:
            m = input("👨‍🍳 How many meals per day? (default 4): ").strip()
            state.meals_per_day = int(m) if m else 4

        shopping_list_manager("clear")

        # Normalize to AgentState before invoking LangGraph
        if isinstance(state, dict):
            state = AgentState(**state)

        state = app.invoke(state)  # run LangGraph
        save_memory(state)


if __name__ == "__main__":
    run_cli()
