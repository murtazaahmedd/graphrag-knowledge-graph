# Usage Service

The Usage Service collects metered events from customer workspaces and aggregates them per hour.
The Usage Service is owned by the Platform Team.
The Usage Service depends on the Event Gateway.
The Usage Service stores data in BigQuery.
The Usage Service runs on Cloud Run.
Late events up to 48 hours old are accepted and re-aggregated.
