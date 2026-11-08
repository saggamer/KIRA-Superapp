import unittest

from scripts.prepare_kira_agentic_stage1 import is_mutating_tool


class AgenticDataPermissionPolicyTests(unittest.TestCase):
    def test_settings_navigation_does_not_require_confirmation(self):
        self.assertFalse(is_mutating_tool("open_wifi_settings"))

    def test_sensitive_changes_require_confirmation(self):
        for name in ("send_email", "create_contact", "create_calendar_event", "delete_file", "set_security_permission"):
            with self.subTest(name=name):
                self.assertTrue(is_mutating_tool(name))


if __name__ == "__main__":
    unittest.main()
