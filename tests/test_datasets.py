from __future__ import annotations

import io
import json
import unittest
from pathlib import Path

from sdk.core.types import MemoryEvent
from scripts import synthetic_events


DATASET_DIR = Path(__file__).resolve().parents[1] / "knowledge" / "datasets"


class TestDatasets(unittest.TestCase):
    def test_gold_datasets_exist(self):
        for name in ["qa.jsonl", "entities.jsonl", "snippets.jsonl"]:
            path = DATASET_DIR / name
            self.assertTrue(path.exists(), f"Expected dataset {name} to exist")
            with path.open("r", encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh if line.strip()]
            self.assertGreaterEqual(len(lines), 1)

    def test_synthetic_event_generator_respects_catch22(self):
        events = synthetic_events.generate_events(5)
        self.assertEqual(len(events), 5)
        for event in events:
            self.assertGreaterEqual(event.score_total, 22.0)
            self.assertTrue(event.embedding)

    def test_cli_serialises_events(self):
        events = synthetic_events.generate_events(3)
        payload = "\n".join(json.dumps(e.to_dict()) for e in events)
        buffer = io.StringIO()
        buffer.write(payload)
        buffer.seek(0)
        for line in buffer:
            data = json.loads(line)
            restored = MemoryEvent.from_dict(data)
            self.assertEqual(restored.id, data["id"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

