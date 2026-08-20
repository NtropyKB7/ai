# ChromaDB persistence

The `web` service stores Chroma data outside the container through a bind
mount. Docker Compose resolves the host path, while the Python application
uses a fixed container path.

| Variable | Owner | Development example | Production example |
| --- | --- | --- | --- |
| `CHROMA_HOST_DIR` | Docker Compose host | `./.local/chroma-data` | `/home/ubuntu/chroma-data/financial_products_finlife_v20260820_issue38` |
| `CHROMA_DB_DIR` | Application container | `/data/chroma` | `/data/chroma` |
| `FINANCIAL_PRODUCT_COLLECTION` | Application container | `financial_products` | `financial_products_finlife_v20260820_issue38` |

The defaults in `docker-compose.yml` keep local data under the ignored
`.local/` directory. Running the application directly, outside Compose,
continues to use the `Settings.CHROMA_DB_DIR` default of `./.chroma` unless a
local environment overrides it.

## Production prerequisites

Before recreating the container:

1. Place the validated Chroma persistent directory on the existing EBS.
2. Verify that the directory contains `chroma.sqlite3` and that its active
   collection exists with the expected count.
3. Add the three variables above to the existing GitHub Actions `ENV_FILE`
   secret. Never commit the production `.env` or any API key.
4. Run `docker-compose config` and confirm the resolved bind source without
   publishing the full output, because it may contain environment values.

The deployment workflow rewrites `/home/ubuntu/app/.env` from `ENV_FILE` on
every `develop` deployment. A manual EC2-only edit is therefore not durable.

## Read-only verification

Confirm the mount after deployment without printing product data:

```bash
docker inspect fastapi-app \
  --format '{{range .Mounts}}{{println .Type .Source "->" .Destination}}{{end}}'

docker exec fastapi-app python -c \
  'from app.core.config import settings; print(settings.CHROMA_DB_DIR, settings.FINANCIAL_PRODUCT_COLLECTION)'

docker exec fastapi-app python -c \
  'from app.rag.chroma_client import chroma_manager; c=chroma_manager.client.get_collection(chroma_manager.collection_name); print(c.count())'
```

Expected mount destination: `/data/chroma`. Do not run the collection command
until the transferred database and installed Chroma version have been
validated.

## Container recreation persistence check

Use only a non-production test directory for this check:

1. Start Compose with `CHROMA_HOST_DIR` pointing at the test directory.
2. Add a synthetic record to a test-only collection.
3. Run `docker-compose down` and `docker-compose up -d`.
4. Confirm that the synthetic collection and record still exist.
5. Remove only the test directory after the check; never delete an operating
   or versioned serving directory.

## Rollback

Keep the previous persistent directory unchanged. To roll back, restore both
of these values in the GitHub Actions `ENV_FILE` secret:

```text
CHROMA_HOST_DIR=<previous EBS Chroma directory>
FINANCIAL_PRODUCT_COLLECTION=<previous collection name>
```

Keep `CHROMA_DB_DIR=/data/chroma`, then recreate only the `web` service. Do not
delete either versioned directory during rollback.
