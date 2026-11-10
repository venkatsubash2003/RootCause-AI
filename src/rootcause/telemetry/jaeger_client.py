
import os
import requests

JAEGER_BASE_URL = os.getenv(
    "JAEGER_BASE_URL",
    "http://localhost:63295/jaeger/ui"
)


def get_services():
    url = f"{JAEGER_BASE_URL}/api/v3/services"

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()

    return data.get("services", [])


if __name__ == "__main__":
    services = get_services()

    print(f"Total services: {len(services)}")

    for service in services:
        print(f"- {service}")
