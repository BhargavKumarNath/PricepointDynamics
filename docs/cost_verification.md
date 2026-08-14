# Cost Verification

`project_refactor.md` §18/§25.9 claim this architecture is realistically
£0/month at portfolio traffic. This page is a real verification pass
against that claim — what was actually re-checked, what's still true,
and, honestly, what **cannot** be verified from this environment.

## What this pass actually checked (2026-08-14)

Re-fetched each provider's current official pricing/docs page rather
than trusting the numbers already written into §18/§25.9 (those were
last verified 2026-08-12 — two days is short, but the discipline is the
point: re-verify, don't assume a decision stays true because it was true
once).

| Claim | Source checked | Result |
|---|---|---|
| Vercel Hobby is free but restricted to personal/non-commercial use | vercel.com/pricing | **Confirmed.** Also newly noted: 100GB/mo Fast Data Transfer, 1M edge requests/mo, 1M serverless invocations/mo, 4 CPU-hours/mo active compute — all moot for this project since the frontend is pure static export with no serverless functions in the request path. |
| Cloud Run "Always Free": 2M requests/month, billed by request-processing time, scale-to-zero, requires a billing account on file | cloud.google.com/run/pricing | **Confirmed** on all points above. The exact vCPU-second/GiB-second split (§25.5 states 180,000 vCPU-s + 360,000 GiB-s) **could not be confidently re-verified** — two automated fetch attempts of the same live page returned the two numbers in different order, and the page's full pricing table didn't come through cleanly either time. The qualitative facts that actually drive the Cloud-Run-over-Render decision (request-time billing, true scale-to-zero, 2M free requests) are solid; the precise compute-second ceiling is not re-confirmed here — check `cloud.google.com/run/pricing` directly before relying on the exact number. |
| Cloudflare R2: 10GB free storage, 1M Class A + 10M Class B ops/month, zero egress fees | developers.cloudflare.com/r2/pricing | **Confirmed** on storage/ops/egress. Whether a card is required to activate R2 even while staying free (§25.9's existing hedge, "likely needs a card") **remains unconfirmed** — the current pricing page doesn't state this either way; this hedge is carried forward unchanged, not newly verified. |
| GitHub Actions: free/unlimited minutes on public repos; private repos get a metered free allowance (2,000 min/mo on the Free plan) | docs.github.com/en/billing/concepts/product-billing/github-actions | **Confirmed.** GitHub Pages hosting is free with no stated storage limit in the fetched content. |

## What could not be verified: actual live spend

**Nothing in this project is currently deployed.** No Vercel project, no
Cloud Run service, no R2 bucket has been provisioned — see
`docs/deployment.md` for exactly what each would require (credentials
and, in most cases, a billing account this development environment
doesn't have and shouldn't provision unilaterally, per
`project_refactor.md`'s own precedent set in Phase 5).

This means the honest answer to "confirm actual monthly spend is £0" is:
**there is no live spend to measure, because there is no live
deployment.** The verification this page performs instead is the
closest available substitute — confirming the *free-tier claims
themselves* are still accurate against each provider's current terms,
so that a future deployer can trust the numbers in §18/§25.9 and
`docs/deployment.md` at the moment they actually do deploy. Re-run this
check again at deploy time; pricing pages change.

## Repo-visibility caveat (GitHub Actions minutes)

The GitHub Actions free-minutes claim above depends on whether this
repository is public or private — public repos get unlimited free
minutes, private repos get a metered 2,000 min/mo allowance (Free plan).
This could not be confirmed from this environment: no `gh` CLI is
installed, and an unauthenticated fetch of the repo's GitHub URL returned
a 404 (consistent with either a private repo or simply an outdated URL —
not distinguishable from here). **Check `Settings → General → Danger
Zone` on the actual repo, or `gh repo view --json visibility`, before
relying on "CI is free" as a settled fact.**

## The one accepted non-zero cost

A custom domain (~£8-12/year) remains the one deliberate, optional,
non-zero recurring cost if desired — the free `*.vercel.app` subdomain
keeps the deployment genuinely £0/month otherwise.
