"""Run in the deployed Render shell. No real API keys are sent to test hosts."""
import argparse
import importlib.metadata as metadata
import json
import os
import ssl
import sys

import certifi
import requests
from google import genai
from openai import OpenAI

BAD_HOSTS = (
    "https://dv-g3b-emudhra.we-have-a-problem.com/",
    "https://dv-g3b-aaa-emudhra.we-have-a-problem.com/",
)
CA_OVERRIDES = ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR")


def certificate_error(error):
    """Only certificate verification failures count; DNS/timeouts do not."""
    pending, seen = [error], set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, ssl.SSLCertVerificationError):
            return True
        if isinstance(value, BaseException):
            pending.extend((value.__cause__, value.__context__))
            pending.extend(value.args)
            pending.append(getattr(value, "reason", None))
        # Some SDKs/urllib3 retain only a string representation of the SSL error.
        if isinstance(value, str) and "CERTIFICATE_VERIFY_FAILED" in value:
            return True
    return False


def probe(client, url):
    if client == "requests":
        with requests.get(url, timeout=(5, 10), allow_redirects=False, stream=True) as response:
            return response.status_code
    if client == "openai":
        # Same default SDK transport as main.get_openai_client; no production key.
        with OpenAI(api_key="tls-probe-not-a-secret", base_url=url,
                    timeout=10, max_retries=0) as sdk:
            sdk.models.list()
        return 200
    # main uses the synchronous Gemini client, including in asyncio.to_thread.
    with genai.Client(vertexai=False, api_key="tls-probe-not-a-secret", http_options={
        "base_url": url, "api_version": "v1beta", "timeout": 10000,
    }) as sdk:
        next(iter(sdk.models.list()), None)
    return 200


def rejection_check(client, url):
    try:
        status = probe(client, url)
    except Exception as error:
        if certificate_error(error):
            return "PASS_CERTIFICATE_REJECTED"
        # HTTP errors also prove that the TLS handshake succeeded.
        if getattr(error, "status_code", None) or getattr(error, "code", None):
            return "FAIL_TLS_ACCEPTED"
        return "INCONCLUSIVE_" + type(error).__name__
    return "FAIL_TLS_ACCEPTED_HTTP_" + str(status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", action="store_true",
                        help="Test the two public hosts from Render's advisory")
    args = parser.parse_args()
    versions = {name: metadata.version(name) for name in (
        "certifi", "pip", "requests", "urllib3", "openai", "google-genai",
        "fastapi", "starlette", "python-multipart",
    )}
    ca_ok = tuple(map(int, versions["certifi"].split("."))) >= (2026, 10, 19)
    pip_ok = tuple(map(int, versions["pip"].split(".")[:3])) >= (26, 2, 2)
    overrides = [name for name in CA_OVERRIDES if os.environ.get(name)]
    report = {"python": sys.version.split()[0], "openssl": ssl.OPENSSL_VERSION,
              "versions": versions, "certifi_path": certifi.where(),
              "ca_overrides_set": overrides,
              "proxy_configured": any(os.environ.get(key) for key in (
                  "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")),
              "certifi_minimum_met": ca_ok, "pip_minimum_met": pip_ok}
    passed = ca_ok and pip_ok and not overrides
    if args.network:
        results = []
        for client in ("requests", "openai", "gemini"):
            for url in BAD_HOSTS:
                result = rejection_check(client, url)
                results.append({"client": client, "url": url, "result": result})
                passed = passed and result == "PASS_CERTIFICATE_REJECTED"
        report["rejection_tests"] = results
    report["scope"] = "runtime TLS probes" if args.network else "offline versions only; TLS not tested"
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
