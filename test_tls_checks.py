import ssl
import unittest
from unittest.mock import patch

from scripts.check_tls import certificate_error, rejection_check


class TLSChecksTest(unittest.TestCase):
    def test_timeout_and_dns_are_not_certificate_rejection(self):
        for error in (TimeoutError("timeout"), OSError("DNS unavailable")):
            self.assertFalse(certificate_error(error))
            with patch("scripts.check_tls.probe", side_effect=error):
                self.assertTrue(rejection_check("requests", "https://example.test").startswith("INCONCLUSIVE"))

    def test_wrapped_certificate_failure_is_recognized(self):
        error = RuntimeError("transport failed")
        error.__cause__ = ssl.SSLCertVerificationError(1, "CERTIFICATE_VERIFY_FAILED")
        with patch("scripts.check_tls.probe", side_effect=error):
            self.assertEqual(rejection_check("openai", "https://example.test"), "PASS_CERTIFICATE_REJECTED")

    def test_any_http_response_is_failed_rejection(self):
        with patch("scripts.check_tls.probe", return_value=403):
            self.assertEqual(rejection_check("requests", "https://example.test"), "FAIL_TLS_ACCEPTED_HTTP_403")


if __name__ == "__main__":
    unittest.main()
