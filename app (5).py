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



# API KEY


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured.")



# TOOL 1 - WEATHER


@tool
def get_weather(city: str) -> str:
    """Get current temperature and weather information for a city."""

    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_params = {
            "name": city,
            "count": 1
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        ).json()

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

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        ).json()

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
def search_places_and_hotels(
    city: str,
    category: str = "all"
) -> str:
    """
    Find tourist attractions and hotels.
    category can be hotels, attractions, or all.
    """

    travel_db = {

        "paris": {
            "attractions": [
                "Eiffel Tower",
                "Louvre Museum",
                "Arc de Triomphe"
            ],
            "hotels": [
                "Le Meurice",
                "Hotel Plaza Athénée",
                "Ritz Paris"
            ]
        },

        "tokyo": {
            "attractions": [
                "Senso-ji Temple",
                "Tokyo Tower",
                "Shibuya Crossing"
            ],
            "hotels": [
                "Aman Tokyo",
                "Park Hyatt Tokyo",
                "Keio Plaza Hotel"
            ]
        },

        "mumbai": {
            "attractions": [
                "Gateway of India",
                "Marine Drive",
                "Elephanta Caves"
            ],
            "hotels": [
                "The Taj Mahal Palace",
                "The Oberoi",
                "JW Marriott Juhu"
            ]
        },

        "new york": {
            "attractions": [
                "Statue of Liberty",
                "Central Park",
                "Times Square"
            ],
            "hotels": [
                "The Plaza",
                "The Ritz-Carlton Central Park",
                "Ace Hotel"
            ]
        }
    }

    city_key = city.lower().strip()

    city_data = travel_db.get(city_key)

    if not city_data:
        return (
            f"Top recommendations in {city}: "
            "Central Sightseeing Tour, Grand Hotel, "
            "City Heritage Museum."
        )

    category = category.lower()

    if category == "hotels":

        return json.dumps({
            "city": city,
            "recommended_hotels": city_data["hotels"]
        })

    elif category == "attractions":

        return json.dumps({
            "city": city,
            "tourist_places": city_data["attractions"]
        })

    else:

        return json.dumps({
            "city": city,
            "recommendations": city_data
        })



# TOOL 3 - CURRENCY CONVERTER


@tool
def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str
) -> str:
    """Convert money between currencies."""

    try:

        from_currency = from_currency.upper()
        to_currency = to_currency.upper()

        url = (
            f"https://open.er-api.com/v6/latest/"
            f"{from_currency}"
        )

        response = requests.get(
            url,
            timeout=10
        ).json()

        if response.get("result") != "success":
            return "Unable to retrieve exchange rate."

        rates = response.get("rates", {})

        rate = rates.get(to_currency)

        if not rate:
            return (
                f"Currency {to_currency} "
                "was not found."
            )

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



# THREE TOOLS

travel_tools = [
    get_weather,
    search_places_and_hotels,
    convert_currency
]



# GEMINI LLM

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0.3
)



# SMART TRAVEL AGENT

smart_travel_agent = create_agent(
    model=llm,
    tools=travel_tools,

    system_prompt=(
        "You are Smart Travel Agent, "
        "an expert travel planning assistant. "

        "You have three tools: "

        "1. get_weather for current weather. "
        "2. search_places_and_hotels for hotels "
        "and tourist attractions. "
        "3. convert_currency for currency conversion. "

        "Use the appropriate tools when required. "
        "You may use multiple tools for one question. "

        "Always provide clear and concise answers. "
        "Never invent weather or exchange-rate data."
    )
)



# REQUEST MODEL


class UserQuery(BaseModel):
    query: str



# CHAT ENDPOINT

@app.post("/chat")
def chat(request: UserQuery):

    try:

        result = smart_travel_agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": request.query
                }
            ]
        })

        answer = result["messages"][-1].content

        if isinstance(answer, list):

            answer = "".join(
                item.get("text", "")
                if isinstance(item, dict)
                else str(item)
                for item in answer
            )

        return {
            "response": str(answer)
        }

    except Exception as e:

        return {
            "error": str(e)
        }



# HOME - SIMPLE CHATBOT UI

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>

<html>

<head>
    <title>Smart Travel Agent</title>
</head>

<body>

    <h2>🌍 Smart Travel Agent</h2>

    <p>Ask me about weather, hotels, tourist places, or currency conversion.</p>

    <hr>

    <div id="chat"></div>

    <br>

    <input
        type="text"
        id="message"
        placeholder="Ask about your travel..."
        style="width: 400px;"
    >

    <button onclick="sendMessage()">
        Send
    </button>


<script>

async function sendMessage() {

    const input = document.getElementById("message");

    const message = input.value.trim();

    if (!message) {
        return;
    }

    const chat = document.getElementById("chat");


    // Show user message

    chat.innerHTML +=
        "<p><b>You:</b> " +
        message +
        "</p>";


    // Clear input

    input.value = "";


    // Show thinking message

    chat.innerHTML +=
        "<p><b>Agent:</b> Thinking...</p>";


    try {

        const response = await fetch("/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                query: message
            })

        });


        const data = await response.json();


        const messages =
            chat.getElementsByTagName("p");


        const lastMessage =
            messages[messages.length - 1];


        if (data.response) {

            lastMessage.innerHTML =
                "<b>Agent:</b> " +
                data.response;

        }

        else {

            lastMessage.innerHTML =
                "<b>Agent:</b> Error: " +
                (data.error || "Something went wrong.");

        }

    }

    catch (error) {

        const messages =
            chat.getElementsByTagName("p");


        messages[messages.length - 1].innerHTML =
            "<b>Agent:</b> Connection error.";

    }

}



// PRESS ENTER TO SEND

document
    .getElementById("message")
    .addEventListener(
        "keydown",

        function(event) {

            if (event.key === "Enter") {

                sendMessage();

            }

        }
    );

</script>

</body>

</html>
"""



# HEALTH CHECK

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }
