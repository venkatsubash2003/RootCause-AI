
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from rootcause.telemetry.trace_normalizer import (
    normalize_traces
)


def test_normalize_traces():

    sample = {
        "result": {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {
                                    "stringValue": "checkout"
                                }
                            }
                        ]
                    },
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": "test-trace",
                                    "spanId": "span-1",
                                    "name": "PlaceOrder",
                                    "startTimeUnixNano":
                                        "1000000000",
                                    "endTimeUnixNano":
                                        "1104000000",
                                    "status": {"code": 1}
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    }

    result = normalize_traces(sample)

    assert len(result) == 1
    assert result[0]["service"] == "checkout"
    assert result[0]["operation"] == "PlaceOrder"
    assert result[0]["duration_ms"] == 104.0
    assert result[0]["status"] == 1
