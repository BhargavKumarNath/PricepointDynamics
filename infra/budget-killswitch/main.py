"""Cloud Function (gen2) — disables billing on the project once a Cloud
Billing budget reports actual spend above the budget amount.

Triggered by Pub/Sub messages published by the budget
`PricePoint monthly cap (10 GBP)` to the topic `billing-budget-alerts`.

The runtime service account (budget-killswitch@…) holds only
roles/billing.projectManager on this project — enough to detach the
billing account, nothing else.
"""

from __future__ import annotations

import base64
import json
import os

import functions_framework
import google.auth
from googleapiclient import discovery

PROJECT_ID = os.environ.get("BILLING_PROJECT_ID", "pricepoint-dynamics")
PROJECT_NAME = f"projects/{PROJECT_ID}"


@functions_framework.cloud_event
def stop_billing(cloud_event) -> None:
    try:
        raw = base64.b64decode(cloud_event.data["message"]["data"]).decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"could not decode Pub/Sub message: {exc!r}")
        return

    data = json.loads(raw)
    cost = float(data.get("costAmount", 0) or 0)
    budget = float(data.get("budgetAmount", 0) or 0)
    print(
        f"budget='{data.get('budgetDisplayName')}' costAmount={cost} "
        f"budgetAmount={budget} thresholdExceeded={data.get('alertThresholdExceeded')}"
    )

    if budget <= 0 or cost <= budget:
        print("under budget — no action")
        return

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    billing = discovery.build("cloudbilling", "v1", credentials=credentials, cache_discovery=False)

    info = billing.projects().getBillingInfo(name=PROJECT_NAME).execute()
    if not info.get("billingEnabled"):
        print("billing already disabled — nothing to do")
        return

    billing.projects().updateBillingInfo(name=PROJECT_NAME, body={"billingAccountName": ""}).execute()
    print(f"BILLING DISABLED for {PROJECT_NAME} (cost {cost} > budget {budget})")
