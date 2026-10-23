import json
import unittest

from diagonal_browser.result_projection import render_browser_result


class BrowserProjectionTests(unittest.TestCase):
    def test_history_pagination_advances_only_past_entries_actually_shown(self):
        result = json.loads(render_browser_result({
            'total': 50, 'offset': 0, 'nextOffset': None,
            'entries': [{'historyId': f'entry-{index}', 'title': 'A saved source', 'snippet': 'Evidence ' * 40}
                        for index in range(50)],
        }))
        self.assertEqual(len(result['entries']), 10)
        self.assertEqual(result['nextOffset'], 10)
        self.assertEqual(result['omittedEntries'], 40)

    def test_large_view_remains_valid_json_with_actionable_controls(self):
        view = {
            "tabId": "tab-a", "url": "https://example.com", "observationId": "obs-a",
            "text": "article " * 3000,
            "elements": [
                {"index": index, "role": "link", "name": f"Link {index}",
                 "href": f"https://example.com/{index}", "x": 100, "y": 200,
                 "gridCell": "B2", "bounds": {"x": 90, "y": 190, "width": 20, "height": 20}}
                for index in range(80)
            ],
        }
        rendered = render_browser_result(view)
        result = json.loads(rendered)
        self.assertLessEqual(len(rendered), 16000)
        self.assertEqual(result["observationId"], "obs-a")
        self.assertEqual(result["elements"][0]["index"], 0)
        self.assertEqual(result["elements"][0]["href"], "https://example.com/0")
        self.assertIn("textOmittedChars", result)


if __name__ == "__main__":
    unittest.main()
