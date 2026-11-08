import logging
import unittest
from scripts.colab_safe import CredentialFilter


class ColabLogTests(unittest.TestCase):
    def test_credentials_are_removed_from_formatted_logs(self):
        for template, args in (
            ('Request failed with Bearer %s', ('fake-sensitive-token',)),
            ('authorization: %s', ('fake-sensitive-token',)),
            ('refresh_token=%s', ('fake-sensitive-token',)),
            ('failure ya29.fake-sensitive-token', ()),
            ('request failed: https://example.invalid/?colab-runtime-proxy-token=%s', ('fake-sensitive-token',)),
        ):
            record = logging.LogRecord('client', logging.WARNING, '', 1, template, args, None)
            self.assertTrue(CredentialFilter().filter(record))
            self.assertNotIn('fake-sensitive-token', record.getMessage())
        record = logging.LogRecord('client', logging.WARNING, '', 1, 'training failed: timeout', (), None)
        CredentialFilter().filter(record)
        self.assertEqual(record.getMessage(), 'training failed: timeout')
