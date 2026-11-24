# Incident notes

In March the Revenue Dashboard showed zero revenue for six hours. The root cause was an expired credential in the
Identity Service, which made the Billing Service fail its authentication checks.
In June the Usage Dashboard lagged by a day because the Event Gateway dropped events during a Pub/Sub quota spike.
Runbooks for both incidents live in the Platform Team wiki.
