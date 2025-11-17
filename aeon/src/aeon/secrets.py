import os
from typing import Optional

from infisical_sdk import InfisicalSDKClient

from aeon.logging import logger


class SecretManager:


    def __init__(
        self,
        client_secret: Optional[str] = None,
        client_id: str = "fc948243-80a0-4831-9e2d-606e473d82c8",
        project_slug: str = "aeon-la-qo",
    ):
        self.client_secret = client_secret or os.environ.get("INFISICAL_CLIENT_SECRET")
        if self.client_secret is None:
            raise RuntimeError(
                "client_secret must be provided explicitly or exposed as an env var."
            )
        self.client_id = client_id
        self.project_slug = project_slug
        self.client = InfisicalSDKClient(host="https://app.infisical.com")
        self.client.auth.universal_auth.login(
            client_id=self.client_id,
            client_secret=self.client_secret
        )

    def get_secrets(self, env: str = "dev", secret_path: str = "/") -> dict:
        """
        Returns
        -------
        Dict mapping secret name to secret value.
        """
        raw = self.client.secrets.list_secrets(
            project_slug=self.project_slug,
            environment_slug=env,
            secret_path=secret_path
        )
        return {secret.secretKey: secret.secretValue for secret in raw.secrets}

    def set_secrets(self):
        """Set secrets as env vars."""
        for k, v in self.get_secrets().items():
            os.environ[k] = v
