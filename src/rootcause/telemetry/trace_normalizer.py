
def extract_attributes(attributes):
    """
    Convert OpenTelemetry attributes into a dictionary.
    """
    result = {}

    for item in attributes:
        key = item.get("key")
        value = item.get("value", {})

        if key:
            if isinstance(value, dict):
                result[key] = next(
                    iter(value.values()),
                    None
                )
            else:
                result[key] = value

    return result


attributes = [
    {
        "key": "service.name",
        "value": {"stringValue": "checkout"}
    }
]

print(extract_attributes(attributes))



def normalize_traces(payload):
    """
    Transform Jaeger v3 trace JSON into
    structured span records.
    """

    result = payload.get("result", payload)

    resource_spans = result.get("resourceSpans", [])

    records = []

    for resource_group in resource_spans:

        resource = resource_group.get("resource", {})

        resource_attributes = extract_attributes(
            resource.get("attributes", [])
        )

        service_name = resource_attributes.get(
            "service.name",
            "unknown"
        )

        for scope in resource_group.get(
            "scopeSpans", []
        ):

            for span in scope.get("spans", []):

                start = int(
                    span.get("startTimeUnixNano", 0)
                )

                end = int(
                    span.get("endTimeUnixNano", 0)
                )

                duration_ms = (
                    (end - start) / 1_000_000
                    if start > 0 and end >= start
                    else None
                )

                status = span.get("status", {})

                records.append({
                    "trace_id": span.get("traceId"),
                    "span_id": span.get("spanId"),
                    "parent_span_id": span.get(
                        "parentSpanId"
                    ),
                    "service": service_name,
                    "operation": span.get("name"),
                    "duration_ms": duration_ms,
                    "status": status.get("code", 0),
                })

    return records
