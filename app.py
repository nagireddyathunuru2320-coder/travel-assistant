import os
import json
import requests

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

# FASTAPI APP
app = FastAPI(
    title="Smart Travel Agent",
    description="AI Travel Agent using Gemini and 3 tools"
)

# API KEY & CONFIGURATION
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# TOOL 1 - WEATHER
@tool
def get_weather(city: str) -> str:
    """Get current temperature and weather information for a city."""
    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": city, "count": 1}

        geo_response = requests.get(geo_url, params=geo_params, timeout=10).json()
        if "results" not in geo_response:
            return f"Could not find coordinates for {city}."

        location = geo_response["results"][0]
        latitude = location["latitude"]
        longitude = location["longitude"]

        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "temperature_unit": "celsius"
        }

        weather_response = requests.get(weather_url, params=weather_params, timeout=10).json()
        current = weather_response["current"]

        result = {
            "city": location["name"],
            "country": location.get("country", ""),
            "temperature_celsius": current["temperature_2m"],
            "weather_code": current["weather_code"]
        }
        return json.dumps(result)
    except Exception as e:
        return f"Weather error: {str(e)}"

# TOOL 2 - PLACES AND HOTELS
@tool
def search_places_and_hotels(city: str, category: str = "all") -> str:
    """Find tourist attractions and hotels. category can be hotels, attractions, or all."""
    travel_db = {
        "paris": {
            "attractions": ["Eiffel Tower", "Louvre Museum", "Arc de Triomphe"],
            "hotels": ["Le Meurice", "Hotel Plaza Athénée", "Ritz Paris"]
        },
        "tokyo": {
            "attractions": ["Senso-ji Temple", "Tokyo Tower", "Shibuya Crossing"],
            "hotels": ["Aman Tokyo", "Park Hyatt Tokyo", "Keio Plaza Hotel"]
        },
        "mumbai": {
            "attractions": ["Gateway of India", "Marine Drive", "Elephanta Caves"],
            "hotels": ["The Taj Mahal Palace", "The Oberoi", "JW Marriott Juhu"]
        },
        "new york": {
            "attractions": ["Statue of Liberty", "Central Park", "Times Square"],
            "hotels": ["The Plaza", "The Ritz-Carlton Central Park", "Ace Hotel"]
        }
    }

    city_key = city.lower().strip()
    city_data = travel_db.get(city_key)

    if not city_data:
        return (
            f"Top recommendations in {city}: "
            "Central Sightseeing Tour, Grand Hotel, City Heritage Museum."
        )

    category = category.lower()
    if category == "hotels":
        return json.dumps({"city": city, "recommended_hotels": city_data["hotels"]})
    elif category == "attractions":
        return json.dumps({"city": city, "tourist_places": city_data["attractions"]})
    else:
        return json.dumps({"city": city, "recommendations": city_data})

# TOOL 3 - CURRENCY CONVERTER
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert money between currencies."""
    try:
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()

        url = f"https://open.er-api.com/v6/latest/{from_currency}"
        response = requests.get(url, timeout=10).json()

        if response.get("result") != "success":
            return "Unable to retrieve exchange rate."

        rates = response.get("rates", {})
        rate = rates.get(to_currency)

        if not rate:
            return f"Currency {to_currency} was not found."

        converted = round(amount * rate, 2)
        return json.dumps({
            "original_amount": amount,
            "from_currency": from_currency,
            "converted_amount": converted,
            "to_currency": to_currency,
            "exchange_rate": rate
        })
    except Exception as e:
        return f"Currency conversion error: {str(e)}"

travel_tools = [
    get_weather,
    search_places_and_hotels,
    convert_currency
]

# AGENT FACTORY HELPER (Lazy initialization ensures Render starts even during builds)
_agent = None

def get_travel_agent():
    global _agent
    if _agent is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing on Render.")

        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=api_key,
            temperature=0.3
        )

        _agent = create_agent(
            model=llm,
            tools=travel_tools,
            system_prompt=(
                "You are Smart Travel Agent, an expert travel planning assistant. "
                "You have three tools: get_weather, search_places_and_hotels, and convert_currency. "
                "Use the appropriate tools when required. Always provide clear and concise answers."
            )
        )
    return _agent

# REQUEST MODEL
class UserQuery(BaseModel):
    query: str

# CHAT ENDPOINT
@app.post("/chat")
def chat(request: UserQuery):
    try:
        agent = get_travel_agent()
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": request.query}
            ]
        })

        answer = result["messages"][-1].content
        if isinstance(answer, list):
            answer = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in answer
            )

        return {"response": str(answer)}
    except Exception as e:
        return {"error": str(e)}

# HOME UI
@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Smart Travel Agent</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 680px; margin: 40px auto; padding: 0 16px; }
        #chat { border: 1px solid #ddd; border-radius: 8px; padding: 16px; min-height: 250px; max-height: 400px; overflow-y: auto; background: #fafafa; }
        .input-bar { display: flex; gap: 8px; margin-top: 12px; }
        input { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 15px; }
        button { padding: 10px 18px; border: none; border-radius: 6px; background: #0066cc; color: white; cursor: pointer; font-size: 15px; }
    </style>
</head>
<body>
    <h2>🌍 Smart Travel Agent</h2>
    <p>Ask about weather, hotels, attractions, or currency exchange.</p>
    <div id="chat"></div>
    <div class="input-bar">
        <input type="text" id="message" placeholder="Ask about your travel..." autocomplete="off">
        <button onclick="sendMessage()">Send</button>
    </div>

<script>
async function sendMessage() {
    const input = document.getElementById("message");
    const message = input.value.trim();
    if (!message) return;

    const chat = document.getElementById("chat");
    chat.innerHTML += "<p><b>You:</b> " + message + "</p>";
    input.value = "";

    const loadingId = "loading-" + Date.now();
    chat.innerHTML += `<p id="${loadingId}"><b>Agent:</b> Thinking...</p>`;
    chat.scrollTop = chat.scrollHeight;

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: message })
        });
        const data = await response.json();
        const loadEl = document.getElementById(loadingId);
        if (data.response) {
            loadEl.innerHTML = "<b>Agent:</b> " + data.response;
        } else {
            loadEl.innerHTML = "<b>Agent:</b> Error: " + (data.error || "Something went wrong.");
        }
    } catch (error) {
        document.getElementById(loadingId).innerHTML = "<b>Agent:</b> Connection error.";
    }
    chat.scrollTop = chat.scrollHeight;
}

document.getElementById("message").addEventListener("keydown", function(e) {
    if (e.key === "Enter") sendMessage();
});
</script>
</body>
</html>"""

# RENDER HEALTH CHECK
@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)