import os, json, re
from typing import TypedDict, Dict, Any, Optional
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()
app = Flask(__name__)

class TravelState(TypedDict):
    message: str
    api_key: str
    extracted: Dict[str, Any]
    plan: Dict[str, Any]
    response: str

def extract_trip_info(state: TravelState) -> Dict[str, Any]:
    text = state["message"].strip()
    low = text.lower()
    dur_m = re.search(r'(\d+)\s*(?:-|to\s*)?(\d+)?\s*(?:day|night|week)', low)
    duration = f"{dur_m.group(1)} days" if dur_m else ("2 days" if "weekend" in low else "3-5 days")
    
    b_m = re.search(r'(?:₹|\$|€|£|rs\.?|inr|usd)\s*([\d,]+)|([\d,]+)\s*(?:inr|rs|rupees|usd|bucks|euros)', low)
    budget = (b_m.group(1) or b_m.group(2)).replace(',', '') if b_m else ""
    curr = "₹" if any(c in low for c in ["₹", "inr", "rs", "rupee"]) else ("€" if "€" in low else ("£" if "£" in low else "$"))

    to_m = re.search(r'(?:trip to|travel to|visit|in|going to)\s+([A-Za-z\s]+?)(?=\s+(?:from|for|with|in|under|on|budget)|\b|$)', text, re.I)
    from_m = re.search(r'from\s+([A-Za-z\s]+?)(?=\s+(?:to|for|with|in|under|on|budget)|\b|$)', text, re.I)
    dest = to_m.group(1).strip() if to_m else ""
    origin = from_m.group(1).strip() if from_m else "Origin city"

    intent = "packing" if "pack" in low else ("food" if any(w in low for w in ["eat", "food", "cuisine"]) else
             "accommodation" if any(w in low for w in ["stay", "hotel", "resort"]) else
             "budget" if any(w in low for w in ["cost", "budget", "how much"]) else
             "timing" if any(w in low for w in ["best time", "season", "when to"]) else "itinerary")
    
    needs_clarification = len(text.split()) < 3 and not dest and any(w in low for w in ["hi", "hello", "hey", "trip", "vacation"])
    return {"extracted": {"destination": dest, "origin": origin, "duration": duration, "budget": budget, "currency": curr, "intent": intent, "needs_clarification": needs_clarification}}

SYSTEM_PROMPT = """You are an elite AI Travel Assistant specializing in practical, realistic travel plans.
Generate a structured travel plan based on the user's request.
Guidelines:
1. Pacing: Realistic schedules, morning/afternoon/evening per day, realistic transit times between stops. Do not overload days.
2. Budget: If specified, allocate across Accommodation (~38%), Food (~25%), Local Transit (~18%), Activities (~12%), Buffer (~7%). Mark all costs as estimates.
3. Food: Curated authentic local dishes and dining spots.
4. Transit & Stay: Practical advice on getting around and best neighborhoods to lodge.
5. Accuracy: NEVER fabricate live booking availability or exact ticket prices. Remind to verify live rates, bookings, and visa rules from official sources.

Output STRICT JSON ONLY (no markdown code blocks, no extra commentary):
{
  "trip_title": "...",
  "destination": "...",
  "duration": "...",
  "overview": "...",
  "daily_itinerary": [{"day": 1, "title": "...", "morning": "...", "afternoon": "...", "evening": "...", "transit": "..."}],
  "budget_breakdown": {"total_estimated": "...", "currency": "...", "accommodation": "...", "food": "...", "transport": "...", "activities": "...", "buffer": "..."},
  "dining_recommendations": ["..."],
  "stay_guidance": "...",
  "transport_tips": "...",
  "packing_essentials": ["..."],
  "important_advisories": ["..."],
  "official_verification_notice": "..."
}"""

def plan_travel(state: TravelState) -> Dict[str, Any]:
    ext = state["extracted"]
    if ext.get("needs_clarification"):
        return {"plan": {
            "trip_title": "Welcome to AI Travel Assistant",
            "overview": "I am ready to craft your personalized travel plan! To get started, please share:\n1. Your desired destination\n2. Trip duration (e.g. 5 days, weekend)\n3. Departure city & approximate budget\n\nExample: 'Plan a 5 day trip to Goa from Hyderabad under ₹30000'",
            "daily_itinerary": [], "budget_breakdown": {}, "dining_recommendations": [], "packing_essentials": [], "important_advisories": []
        }}

    key = state.get("api_key")
    if key:
        for model in ["gemini-2.5-flash", "gemini-1.5-flash"]:
            try:
                llm = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0.3)
                user_msg = f"User Request: {state['message']}\nContext: Dest: {ext['destination']}, Origin: {ext['origin']}, Duration: {ext['duration']}, Budget: {ext['currency']}{ext['budget']}, Intent: {ext['intent']}"
                res = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)])
                raw = re.sub(r"^```json\s*|\s*```$", "", res.content.strip(), flags=re.MULTILINE|re.DOTALL).strip()
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    return {"plan": json.loads(match.group(0))}
            except Exception:
                continue

    dest = ext["destination"] or "Your Destination"
    curr = ext["currency"]
    b_val = int(ext["budget"]) if ext["budget"].isdigit() else (30000 if curr == "₹" else 1500)
    days_m = re.search(r"\d+", ext["duration"])
    days = min(max(int(days_m.group(0)) if days_m else 3, 1), 7)
    
    samples = [
        ("Arrival, check-in, orientation walk", "Historic center & artisan markets", "Sunset viewpoint & welcome coastal dinner", "~25 mins local transit"),
        ("Iconic landmark exploration", "Cultural museum & authentic lunch", "Bustling night market and street food trail", "~20 mins between stops"),
        ("Morning nature hike or scenic coastline", "Local harbor cafe & regional specialties", "Waterfront promenade walk & live music", "~35 mins transit"),
        ("Heritage architecture / temple tour", "Handicrafts & spice shopping", "Rooftop dining with city panoramic views", "~20 mins cab"),
        ("Hidden neighborhood exploration", "Farewell lunch & souvenir shopping", "Departure transfer to transit hub", "~35 mins to airport/station")
    ]
    itinerary = [{"day": i + 1, "title": f"Day {i + 1}: Exploring {dest}", "morning": samples[i % len(samples)][0],
                  "afternoon": samples[i % len(samples)][1], "evening": samples[i % len(samples)][2],
                  "transit": samples[i % len(samples)][3]} for i in range(days)]

    return {"plan": {
        "trip_title": f"{days}-Day Travel Plan: {dest}", "destination": dest, "duration": f"{days} Days",
        "overview": f"A balanced {days}-day journey exploring {dest} from {ext['origin']} tailored to a {curr}{b_val:,} budget.",
        "daily_itinerary": itinerary,
        "budget_breakdown": {
            "total_estimated": f"{curr}{b_val:,}", "currency": curr,
            "accommodation": f"{curr}{int(b_val * 0.38):,} (38%)", "food": f"{curr}{int(b_val * 0.25):,} (25%)",
            "transport": f"{curr}{int(b_val * 0.18):,} (18%)", "activities": f"{curr}{int(b_val * 0.12):,} (12%)",
            "buffer": f"{curr}{int(b_val * 0.07):,} (7% emergency reserve)"
        },
        "dining_recommendations": [f"Authentic local eateries in central {dest}", "Fresh morning breakfast cafes", "Evening seafood / regional specialty diners"],
        "stay_guidance": f"Opt for centrally located stays or boutique guesthouses in {dest} with proximity to transit.",
        "transport_tips": f"Utilize local metro, ride-hailing, or reliable taxis. For transit from {ext['origin']}, book express train/flight in advance.",
        "packing_essentials": ["Comfortable walking shoes", "Weather-appropriate layers", "Universal adapter & power bank", "Valid government ID"],
        "important_advisories": ["Download offline maps and emergency contacts", "Keep modest local cash for street vendors", "Verify local operating hours before visiting"],
        "official_verification_notice": "All prices and schedules are estimates. Please verify live rates, bookings, and visa regulations with official providers."
    }}

def validate_itinerary(state: TravelState) -> Dict[str, Any]:
    plan = state["plan"]
    for d in plan.get("daily_itinerary", []):
        if not d.get("transit"):
            d["transit"] = "~20-30 mins between stops"
    plan["official_verification_notice"] = "⚠️ Official Verification Notice: All pricing, timings, and schedules are approximate planning estimates. Real-time rates, flight/hotel availability, and visa requirements must be verified through official operators and authorities."
    return {"plan": plan}

def structure_response(state: TravelState) -> Dict[str, Any]:
    plan = state["plan"]
    if not plan.get("daily_itinerary"):
        return {"response": plan.get("overview", "")}

    lines = [f"# 🌍 {plan.get('trip_title', 'Travel Plan')}\n", f"{plan.get('overview', '')}\n",
             "## ⏱️ Trip Summary", f"- **Destination**: {plan.get('destination', 'N/A')}", f"- **Duration**: {plan.get('duration', 'N/A')}"]
    b = plan.get("budget_breakdown", {})
    if b and b.get("total_estimated"):
        lines.append(f"- **Estimated Budget**: {b.get('total_estimated')}")
        lines.append("\n### 💰 Estimated Budget Breakdown (Guidance Only)")
        for k in ["accommodation", "food", "transport", "activities", "buffer"]:
            if b.get(k): lines.append(f"- **{k.title()}**: {b.get(k)}")

    lines.append("\n## 📅 Day-by-Day Itinerary")
    for d in plan.get("daily_itinerary", []):
        lines.append(f"\n### {d.get('title', f'Day {d.get('day')}')}")
        lines.append(f"- 🌅 **Morning**: {d.get('morning', '')}")
        lines.append(f"- ☀️ **Afternoon**: {d.get('afternoon', '')}")
        lines.append(f"- 🌙 **Evening**: {d.get('evening', '')}")
        if d.get("transit"): lines.append(f"- 🚗 *Transit Note*: {d.get('transit')}")

    if plan.get("dining_recommendations"):
        lines.append("\n## 🍽️ Dining Recommendations")
        for r in plan["dining_recommendations"]: lines.append(f"- {r}")
    if plan.get("stay_guidance"): lines.append(f"\n## 🏨 Accommodation Guidance\n{plan['stay_guidance']}")
    if plan.get("transport_tips"): lines.append(f"\n## 🚕 Transportation Guidance\n{plan['transport_tips']}")
    if plan.get("packing_essentials"):
        lines.append("\n## 🎒 Packing Essentials")
        for p in plan["packing_essentials"]: lines.append(f"- {p}")
    if plan.get("important_advisories"):
        lines.append("\n## 📌 Important Travel Considerations")
        for adv in plan["important_advisories"]: lines.append(f"- {adv}")

    lines.append(f"\n---\n{plan.get('official_verification_notice', '')}")
    return {"response": "\n".join(lines)}

workflow = StateGraph(TravelState)
workflow.add_node("extract_info", extract_trip_info)
workflow.add_node("plan_travel", plan_travel)
workflow.add_node("validate_itinerary", validate_itinerary)
workflow.add_node("structure_response", structure_response)
workflow.add_edge(START, "extract_info")
workflow.add_edge("extract_info", "plan_travel")
workflow.add_edge("plan_travel", "validate_itinerary")
workflow.add_edge("validate_itinerary", "structure_response")
workflow.add_edge("structure_response", END)
travel_agent = workflow.compile()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Travel Assistant — Intelligent Trip Planner</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root { --bg: #0b0f19; --card: rgba(17,24,39,0.85); --border: rgba(255,255,255,0.08); --primary: #38bdf8; --indigo: #6366f1; --emerald: #10b981; --amber: #f59e0b; --text: #f8fafc; --muted: #94a3b8; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Plus Jakarta Sans', system-ui, sans-serif; background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0b0f19 50%, #05070e 100%); color: var(--text); min-height: 100vh; padding: 24px 16px; }
    .container { max-width: 900px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }
    .card { background: var(--card); backdrop-filter: blur(16px); border: 1px solid var(--border); border-radius: 16px; padding: 22px; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.6); }
    .header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }
    .brand { display: flex; align-items: center; gap: 12px; }
    .logo-icon { width: 44px; height: 44px; background: linear-gradient(135deg, var(--primary), var(--indigo)); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px; box-shadow: 0 0 20px rgba(56,189,248,0.3); }
    .title h1 { font-size: 1.35rem; font-weight: 800; background: linear-gradient(to right, #fff, #93c5fd); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .title p { font-size: 0.85rem; color: var(--muted); }
    .status-badge { display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.25); border-radius: 20px; font-size: 0.75rem; color: var(--emerald); font-weight: 600; }
    .status-dot { width: 7px; height: 7px; background: var(--emerald); border-radius: 50%; box-shadow: 0 0 8px var(--emerald); }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .chip { background: rgba(255,255,255,0.05); border: 1px solid var(--border); border-radius: 20px; padding: 6px 12px; font-size: 0.8rem; cursor: pointer; transition: all 0.2s; color: var(--muted); }
    .chip:hover { background: rgba(56,189,248,0.15); border-color: var(--primary); color: #fff; }
    .input-grid { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 10px; margin-top: 14px; }
    @media(max-width: 680px) { .input-grid { grid-template-columns: 1fr; } }
    input, textarea { width: 100%; background: rgba(0,0,0,0.3); border: 1px solid var(--border); border-radius: 10px; padding: 12px; color: #fff; font-family: inherit; font-size: 0.9rem; outline: none; }
    input:focus, textarea:focus { border-color: var(--primary); box-shadow: 0 0 10px rgba(56,189,248,0.2); }
    textarea { min-height: 85px; resize: vertical; margin-top: 10px; }
    .key-bar { margin-top: 10px; display: flex; gap: 10px; align-items: center; }
    .btn { background: linear-gradient(135deg, var(--primary), var(--indigo)); color: #fff; border: none; border-radius: 10px; padding: 12px 20px; font-weight: 700; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; transition: opacity 0.2s; }
    .btn:hover { opacity: 0.92; }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .alert { padding: 12px 16px; border-radius: 10px; background: rgba(244,63,94,0.15); border: 1px solid rgba(244,63,94,0.3); color: #fca5a5; font-size: 0.85rem; display: none; }
    .loading { display: none; text-align: center; padding: 24px; color: var(--primary); font-size: 0.95rem; }
    .spinner { width: 28px; height: 28px; border: 3px solid rgba(56,189,248,0.2); border-top-color: var(--primary); border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 10px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .result-section { display: none; flex-direction: column; gap: 16px; }
    .user-bubble { background: rgba(56,189,248,0.08); border-left: 3px solid var(--primary); padding: 10px 14px; border-radius: 8px; font-size: 0.85rem; color: #bae6fd; margin-bottom: 12px; }
    .pills { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    .pill { background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.25); color: #7dd3fc; border-radius: 8px; padding: 4px 10px; font-size: 0.8rem; font-weight: 600; }
    .day-card { background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 10px; }
    .day-title { color: var(--primary); font-weight: 700; margin-bottom: 8px; font-size: 1rem; }
    .day-row { font-size: 0.88rem; line-height: 1.5; margin-bottom: 6px; }
    .transit-badge { display: inline-block; background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.2); color: var(--amber); border-radius: 6px; padding: 2px 8px; font-size: 0.75rem; margin-top: 4px; }
    .budget-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-top: 8px; }
    .budget-box { background: rgba(0,0,0,0.25); border: 1px solid var(--border); border-radius: 8px; padding: 10px; text-align: center; }
    .budget-box span { font-size: 0.75rem; color: var(--muted); display: block; }
    .budget-box strong { font-size: 0.95rem; color: #fff; margin-top: 4px; display: block; }
    .notice { background: rgba(245,158,11,0.08); border-left: 3px solid var(--amber); padding: 12px; border-radius: 6px; font-size: 0.8rem; color: #fde68a; line-height: 1.4; margin-top: 10px; }
  </style>
</head>
<body>
<div class="container">
  <div class="card header">
    <div class="brand">
      <div class="logo-icon">✈</div>
      <div class="title">
        <h1>AI Travel Assistant</h1>
        <p>Intelligent Trip Planning & Itinerary Architect</p>
      </div>
    </div>
    <div class="status-badge"><span class="status-dot"></span> LangGraph Ready</div>
  </div>

  <div class="card">
    <label style="font-size:0.85rem; color:var(--muted); font-weight:600;">Trip Prompt & Preferences</label>
    <textarea id="promptInput" placeholder="e.g. Plan a 5 day trip to Goa from Hyderabad under ₹30000 with beaches, seafood, and relaxed vibes"></textarea>
    <div class="input-grid">
      <input type="text" id="destInput" placeholder="Destination (e.g. Goa)">
      <input type="text" id="budgetInput" placeholder="Budget (e.g. ₹30,000)">
      <input type="text" id="datesInput" placeholder="Duration (e.g. 5 Days)">
    </div>
    <div class="key-bar">
      <input type="password" id="apiKeyInput" placeholder="Gemini API Key (optional if GOOGLE_API_KEY is set in environment)" style="font-size:0.8rem; padding:8px 12px;">
    </div>
    <div class="chips">
      <div class="chip" onclick="setPrompt('Plan a 5 day trip to Goa from Hyderabad under ₹30000', 'Goa', '₹30,000', '5 Days')">🏖️ Goa 5 Days (₹30k)</div>
      <div class="chip" onclick="setPrompt('Plan a 4 day cultural trip to Kyoto on $1800', 'Kyoto', '$1,800', '4 Days')">🗼 Tokyo/Kyoto ($1.8k)</div>
      <div class="chip" onclick="setPrompt('Plan a romantic weekend in Paris under €1200', 'Paris', '€1,200', '3 Days')">🥐 Paris Weekend (€1.2k)</div>
      <div class="chip" onclick="setPrompt('Plan a 6 day scenic adventure in Himachal Pradesh under ₹25000', 'Himachal', '₹25,000', '6 Days')">🏔️ Himachal (₹25k)</div>
    </div>
    <div style="margin-top:14px; display:flex; justify-content:flex-end;">
      <button class="btn" id="planBtn" onclick="submitTrip()"><span>Craft My Travel Plan</span> ➔</button>
    </div>
  </div>

  <div id="errorAlert" class="alert"></div>
  <div id="loading" class="loading"><div class="spinner"></div><p id="loadText">Architecting itinerary with LangGraph workflow...</p></div>

  <div id="results" class="card result-section">
    <div id="userQueryBubble" class="user-bubble"></div>
    <div id="pills" class="pills"></div>
    <h2 id="planTitle" style="font-size:1.2rem; color:#fff; margin-bottom:6px;"></h2>
    <div id="planOverview" style="font-size:0.9rem; color:var(--muted); line-height:1.5; margin-bottom:14px;"></div>
    
    <div id="budgetSection" style="margin-bottom:16px;">
      <h3 style="font-size:0.95rem; color:#fff; margin-bottom:6px;">💰 Estimated Budget Breakdown</h3>
      <div id="budgetGrid" class="budget-grid"></div>
    </div>

    <div id="itinerarySection">
      <h3 style="font-size:0.95rem; color:#fff; margin-bottom:10px;">📅 Day-by-Day Schedule</h3>
      <div id="daysContainer"></div>
    </div>

    <div id="tipsSection" style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px;">
      <div class="day-card"><h4 style="color:var(--primary); font-size:0.85rem; margin-bottom:6px;">🍽️ Dining Picks</h4><ul id="diningList" style="font-size:0.82rem; padding-left:16px; color:var(--muted);"></ul></div>
      <div class="day-card"><h4 style="color:var(--primary); font-size:0.85rem; margin-bottom:6px;">🎒 Packing Essentials</h4><ul id="packingList" style="font-size:0.82rem; padding-left:16px; color:var(--muted);"></ul></div>
    </div>

    <div id="guidanceSection" style="margin-top:10px; font-size:0.85rem; color:var(--muted); display:flex; flex-direction:column; gap:6px;">
      <p id="stayText"></p>
      <p id="transitText"></p>
    </div>

    <div id="noticeBox" class="notice"></div>
  </div>
</div>

<script>
function setPrompt(p, d, b, dt) {
  document.getElementById('promptInput').value = p;
  document.getElementById('destInput').value = d;
  document.getElementById('budgetInput').value = b;
  document.getElementById('datesInput').value = dt;
}

async function submitTrip() {
  const pInput = document.getElementById('promptInput');
  const dInput = document.getElementById('destInput').value.trim();
  const bInput = document.getElementById('budgetInput').value.trim();
  const dtInput = document.getElementById('datesInput').value.trim();
  const keyInput = document.getElementById('apiKeyInput').value.trim();
  const btn = document.getElementById('planBtn');
  const alert = document.getElementById('errorAlert');
  const loading = document.getElementById('loading');
  const results = document.getElementById('results');

  let msg = pInput.value.trim();
  if (!msg && dInput) {
    msg = `Plan a trip to ${dInput}` + (dtInput ? ` for ${dtInput}` : '') + (bInput ? ` under ${bInput}` : '');
  }
  if (!msg) {
    alert.innerText = 'Please enter a destination or trip description.';
    alert.style.display = 'block';
    return;
  }

  alert.style.display = 'none';
  results.style.display = 'none';
  loading.style.display = 'block';
  btn.disabled = true;

  try {
    const payload = { message: msg };
    if (keyInput) payload.api_key = keyInput;
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok || data.status !== 'success') {
      throw new Error(data.response || 'Failed to generate travel plan.');
    }
    renderPlan(data.plan, data.response, msg);
  } catch(err) {
    alert.innerText = err.message || 'An error occurred. Please check your request.';
    alert.style.display = 'block';
  } finally {
    loading.style.display = 'none';
    btn.disabled = false;
  }
}

function renderPlan(p, raw, query) {
  const resDiv = document.getElementById('results');
  document.getElementById('userQueryBubble').innerText = '💬 Request: ' + query;
  if (!p || !p.daily_itinerary || p.daily_itinerary.length === 0) {
    document.getElementById('planTitle').innerText = (p && p.trip_title) || 'Travel Advice';
    document.getElementById('planOverview').innerHTML = ((p && p.overview) || raw).replace(/\\n/g, '<br>');
    document.getElementById('budgetSection').style.display = 'none';
    document.getElementById('itinerarySection').style.display = 'none';
    document.getElementById('tipsSection').style.display = 'none';
    document.getElementById('guidanceSection').style.display = 'none';
    document.getElementById('noticeBox').style.display = 'none';
    resDiv.style.display = 'flex';
    return;
  }

  document.getElementById('planTitle').innerText = p.trip_title || 'Custom Travel Plan';
  document.getElementById('planOverview').innerText = p.overview || '';
  
  const pills = document.getElementById('pills');
  pills.innerHTML = '';
  if (p.destination) pills.innerHTML += `<span class="pill">📍 ${p.destination}</span>`;
  if (p.duration) pills.innerHTML += `<span class="pill">⏱️ ${p.duration}</span>`;
  if (p.budget_breakdown && p.budget_breakdown.total_estimated) pills.innerHTML += `<span class="pill">💰 Est. ${p.budget_breakdown.total_estimated}</span>`;

  const bg = document.getElementById('budgetGrid');
  bg.innerHTML = '';
  const bb = p.budget_breakdown || {};
  ['accommodation', 'food', 'transport', 'activities', 'buffer'].forEach(k => {
    if (bb[k]) bg.innerHTML += `<div class="budget-box"><span>${k.toUpperCase()}</span><strong>${bb[k]}</strong></div>`;
  });
  document.getElementById('budgetSection').style.display = bg.innerHTML ? 'block' : 'none';

  const dc = document.getElementById('daysContainer');
  dc.innerHTML = '';
  p.daily_itinerary.forEach(d => {
    dc.innerHTML += `
      <div class="day-card">
        <div class="day-title">${d.title || ('Day ' + d.day)}</div>
        <div class="day-row">🌅 <strong>Morning:</strong> ${d.morning || 'Leisure start'}</div>
        <div class="day-row">☀️ <strong>Afternoon:</strong> ${d.afternoon || 'Local discovery'}</div>
        <div class="day-row">🌙 <strong>Evening:</strong> ${d.evening || 'Relaxation & dining'}</div>
        ${d.transit ? `<div class="transit-badge">🚗 Transit: ${d.transit}</div>` : ''}
      </div>`;
  });
  document.getElementById('itinerarySection').style.display = 'block';

  const dl = document.getElementById('diningList');
  dl.innerHTML = (p.dining_recommendations || []).map(r => `<li>${r}</li>`).join('');
  const pl = document.getElementById('packingList');
  pl.innerHTML = (p.packing_essentials || []).map(r => `<li>${r}</li>`).join('');
  document.getElementById('tipsSection').style.display = (dl.innerHTML || pl.innerHTML) ? 'grid' : 'none';

  document.getElementById('stayText').innerHTML = p.stay_guidance ? `<strong>🏨 Stay Guidance:</strong> ${p.stay_guidance}` : '';
  document.getElementById('transitText').innerHTML = p.transport_tips ? `<strong>🚕 Transit Tips:</strong> ${p.transport_tips}` : '';
  document.getElementById('guidanceSection').style.display = 'flex';

  const nb = document.getElementById('noticeBox');
  nb.innerText = p.official_verification_notice || '';
  nb.style.display = nb.innerText ? 'block' : 'none';
  resDiv.style.display = 'flex';
}
</script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"status": "error", "response": "Invalid request payload. Expected JSON object."}), 400

    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"status": "error", "response": "Trip request message cannot be empty."}), 400

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or data.get("api_key")
    if not api_key:
        return jsonify({"status": "error", "response": "Google Gemini API key is missing. Please set GOOGLE_API_KEY or GEMINI_API_KEY in your environment."}), 500

    try:
        result = travel_agent.invoke({"message": message, "api_key": api_key, "extracted": {}, "plan": {}, "response": ""})
        return jsonify({
            "status": "success",
            "response": result.get("response", ""),
            "plan": result.get("plan", {})
        })
    except Exception:
        return jsonify({"status": "error", "response": "An error occurred while generating your travel plan. Please try again."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
