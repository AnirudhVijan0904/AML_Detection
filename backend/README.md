# Backend

This backend includes a small ML component in `ml/predict.py` which requires Python packages. To enable database-backed historical lookups, set `DB_USE=true` in your `.env` and provide MySQL credentials.

Quick setup for the ML dependencies (in a Python environment):

```bash
python -m pip install -r ml/requirements.txt
```

When `DB_USE=true`, the ML code will query the `transaction` table instead of loading `data/transactions.csv` into memory and will persist prediction rows into the DB (instead of appending to CSV).

Quick API checks (once server + DB are running):

- DB health:

```bash
curl http://localhost:5000/api/debug/db-health
```

- DB info (returns database name and counts):

```bash
curl http://localhost:5000/api/debug/db-info
```

- Manual prediction (example payload):

```bash
curl -s -X POST http://localhost:5000/api/manual/predict \
  -H "Content-Type: application/json" \
  -d '{"fromAccount":"123","toAccount":"456","amount":1000}' | jq
```

