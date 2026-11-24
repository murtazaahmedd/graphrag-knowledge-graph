# Billing Service

The Billing Service turns usage records into monthly invoices.
The Billing Service is owned by the Payments Team.
The Billing Service depends on the Usage Service. The Billing Service depends on the Identity Service.
The Billing Service stores data in Billing Postgres.
The Billing Service runs on Cloud Run.
Invoices are generated on the first day of every month and retried for three days if the payment provider is down.
