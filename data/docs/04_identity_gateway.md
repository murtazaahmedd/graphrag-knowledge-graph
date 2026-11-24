# Identity Service and Event Gateway

The Identity Service manages users, workspaces and API keys.
The Identity Service is owned by the Platform Team.
The Identity Service stores data in Identity Postgres.
The Identity Service runs on GKE.

The Event Gateway receives raw events over HTTPS and publishes them to Pub/Sub.
The Event Gateway is owned by the Platform Team.
The Event Gateway runs on Cloud Run.
