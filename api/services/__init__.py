"""Business logic bridging routers to `pricepoint` (project_refactor.md
§8.3/§25.2). Routers stay thin: parse request, call a service function,
return the response model -- no SQL or model logic in route handlers."""
