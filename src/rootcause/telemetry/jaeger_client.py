
import os
import requests
from datetime import datetime, timedelta, timezone
from trace_normalizer import normalize_traces
from save_spans import save_records

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



def get_traces(service_name, start_time, end_time):
    """
    Retrieve Jaeger traces for a specific UTC time window.

    Args:
        service_name: Target microservice.
        start_time: UTC datetime object.
        end_time: UTC datetime object.

    Returns:
        JSON response returned by Jaeger.
    """

    if start_time.tzinfo is None or end_time.tzinfo is None:
        raise ValueError(
            "Start and end times must be timezone-aware"
        )

    if start_time >= end_time:
        raise ValueError(
            "Start time must be before end time"
        )

    url = f"{JAEGER_BASE_URL}/api/v3/traces"

    params = {
        "query.serviceName": service_name,
        "query.startTimeMin": start_time.astimezone(
            timezone.utc
        ).isoformat(),
        "query.startTimeMax": end_time.astimezone(
            timezone.utc
        ).isoformat(),
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    print("Request URL:", response.url)
    print("HTTP Status:", response.status_code)

    response.raise_for_status()

    return response.json()

    """
    Retrieve recent traces containing a given service.

    Args:
        service_name: Microservice to search.
        minutes: Lookback window in minutes.

    Returns:
        JSON response from Jaeger.
    """

    url = f"{JAEGER_BASE_URL}/api/v3/traces"

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=minutes)

    params = {
        "query.serviceName": service_name,
        "query.startTimeMin": start_time.isoformat(),
        "query.startTimeMax": end_time.isoformat(),
    }

    response = requests.get(
        url,
        params=params,
        timeout=20,
    )

    print("Request URL:", response.url)
    print("HTTP Status:", response.status_code)

    response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    services = get_services()
    print("Available services:", len(services))

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=5)

    traces = get_traces(
        service_name="ad",
        start_time=start_time,
        end_time=end_time
    )

    records = normalize_traces(traces)

    print("Total spans:", len(records))

    for record in records[:5]:
        print(record)
