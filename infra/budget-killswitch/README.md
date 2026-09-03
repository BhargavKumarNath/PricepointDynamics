# Budget kill switch

Hard cap on Google Cloud spend for the `pricepoint-dynamics` project.

The API runs on Cloud Run with scale-to-zero and sits inside the Always
Free tier, so expected spend is ~£0. This is a backstop against a runaway
bill (a bug hammering `/v1/predict`, an accidental `min-instances`, a
misconfigured job): when the Cloud Billing budget reports **actual spend
above the budget amount**, a Cloud Function detaches the billing account
from the project, which stops all billable resources.

## Caveats

- **Not real-time.** Cloud Billing evaluates budgets a few times a day and
  the cost data itself lags by hours. A sudden spike can overshoot the cap
  (realistically caught at £10-15, not exactly £10) before the switch fires.
- **Recovery is manual.** Re-link the billing account in the console
  (Billing -> Account management -> link project), then hit `/health` to
  warm a fresh instance. Cloud Run recovers on the next request; nothing
  needs redeploying.
- Deleting the billing assignment is exactly the outage state the project
  was in on 2026-09-02 (expired free-trial credit). Same symptom:
  `500`/`503` from Google Frontend, then `no available instance`.

## Wiring

```
Budget "PricePoint monthly cap (10 GBP)"   (£10 / calendar month, scoped to this project)
  --> Pub/Sub topic  billing-budget-alerts
        --> Cloud Function (gen2)  budget-killswitch   [europe-west2, python312]
              runs as  budget-killswitch@pricepoint-dynamics.iam.gserviceaccount.com
              which holds  roles/billing.projectManager  on the project
              (enough to detach billing, nothing else)
```

## Reproduce from scratch

```bash
BILLING_ACCOUNT=017068-C487C3-C0F5A8
PROJECT=pricepoint-dynamics
REGION=europe-west2

# APIs
gcloud services enable billingbudgets.googleapis.com pubsub.googleapis.com \
  cloudfunctions.googleapis.com cloudbuild.googleapis.com run.googleapis.com \
  artifactregistry.googleapis.com eventarc.googleapis.com cloudbilling.googleapis.com \
  --project="$PROJECT"

# Budget: £10/month, alerts at 50/90/100% actual + 50/100% forecast,
# publishing to Pub/Sub
gcloud billing budgets create \
  --billing-account="$BILLING_ACCOUNT" \
  --display-name="PricePoint monthly cap (10 GBP)" \
  --budget-amount=10GBP \
  --filter-projects="projects/$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')" \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9 \
  --threshold-rule=percent=1.0 \
  --threshold-rule=percent=0.5,basis=forecasted-spend \
  --threshold-rule=percent=1.0,basis=forecasted-spend

gcloud pubsub topics create billing-budget-alerts --project="$PROJECT"

# Link the budget to the topic (needs the beta surface)
BUDGET_ID=$(gcloud billing budgets list --billing-account="$BILLING_ACCOUNT" \
  --filter="displayName='PricePoint monthly cap (10 GBP)'" --format='value(name.basename())')
gcloud billing budgets update \
  "billingAccounts/$BILLING_ACCOUNT/budgets/$BUDGET_ID" \
  --notifications-rule-pubsub-topic="projects/$PROJECT/topics/billing-budget-alerts"

# Runtime service account + the one role it needs
gcloud iam service-accounts create budget-killswitch \
  --display-name="Budget kill switch (disables billing at cap)" --project="$PROJECT"
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:budget-killswitch@$PROJECT.iam.gserviceaccount.com" \
  --role="roles/billing.projectManager" --condition=None

# Deploy
gcloud functions deploy budget-killswitch \
  --gen2 --runtime=python312 --region="$REGION" \
  --source=. --entry-point=stop_billing \
  --trigger-topic=billing-budget-alerts \
  --service-account="budget-killswitch@$PROJECT.iam.gserviceaccount.com" \
  --set-env-vars=BILLING_PROJECT_ID="$PROJECT" \
  --memory=256Mi --max-instances=1 --timeout=60s

# gen2 + a custom runtime SA: the Eventarc trigger identity also needs these
gcloud run services add-iam-policy-binding budget-killswitch --region="$REGION" \
  --member="serviceAccount:budget-killswitch@$PROJECT.iam.gserviceaccount.com" \
  --role="roles/run.invoker"
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:budget-killswitch@$PROJECT.iam.gserviceaccount.com" \
  --role="roles/eventarc.eventReceiver" --condition=None
```

## Test

Publish a synthetic budget notification. Under budget = no-op (safe):

```bash
gcloud pubsub topics publish billing-budget-alerts --message='{"budgetDisplayName":"PricePoint monthly cap (10 GBP)","costAmount":3.0,"budgetAmount":10.0,"currencyCode":"GBP","alertThresholdExceeded":0.5}'
gcloud functions logs read budget-killswitch --gen2 --region=europe-west2 --limit=10
# expect: "under budget - no action"
```

Do **not** publish a message with `costAmount > budgetAmount` unless you
actually want billing disabled -- that path calls
`updateBillingInfo(billingAccountName="")` for real.
