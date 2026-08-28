# Deployment

Three tiers. The **edge tier is identical in all of them** — a site commissioned
on day one keeps working unchanged as the deployment grows, which is what makes
an incremental rollout possible at all.

| Tier | Hardware | Scope |
|---|---|---|
| Single site | 1 edge node + 1 server | One junction, fully functional standalone |
| Municipal | N edge nodes + 1 server | Kathmandu Valley scale |
| National | N edge nodes + HA Postgres | All seven provinces |

## Single site

```bash
cp deploy/.env.example deploy/.env
```

Generate every secret — none of them have defaults, and the services refuse to
start without them:

```bash
docker compose -f deploy/docker-compose.yml run --rm api scanner-api genkey
```

```bash
docker compose -f deploy/docker-compose.yml up -d
```

Create the first user, then sign in at `http://127.0.0.1:8080`:

```bash
docker compose -f deploy/docker-compose.yml exec api scanner-api adduser --username admin --role admin
```

## What the compose file assumes, and why

**Postgres is not published to the host.** It is reachable only on the compose
network. An exposed Postgres holding a national movement database is the single
worst configuration mistake available here.

**The console binds to loopback.** TLS termination belongs to a reverse proxy in
front of this, using the certificate the deployment actually owns. Shipping a
self-signed certificate would train operators to click through warnings, which
is worse than no TLS story at all.

**Retention runs as its own service, not a thread in the API.** Retention that
shares a process with request handling stops when the API is busy or wedged, and
a legal obligation should not depend on the health of an unrelated component.

**MinIO, not S3.** Data sovereignty is a hard requirement — imagery of Nepali
roads does not leave Nepali infrastructure.

**The API image contains no ML stack.** It never runs inference, so PyTorch,
ONNX Runtime and OpenCV are absent from the most internet-exposed component.
Likewise the edge image contains ONNX Runtime but not PyTorch: a node runs
models, it does not train them.

## `SCANNER_PLATE_KEY` can never change

Every stored plate is indexed by `HMAC-SHA256(key, canonical)`. Rotating the key
orphans the entire history — existing reads become unsearchable and unmatched
against watch-lists. Treat it like a database encryption key: generate once,
store in a KMS or HSM, back it up separately from the database, and never put it
in the same place as the data it protects.

It is also **pseudonymisation, not encryption**. Nepal's plate space is small
enough (2.2 × 10⁸ legacy plates) that anyone holding the key can enumerate it
offline in minutes. It raises the cost of a database disclosure; it does not
make one survivable.

## Edge nodes

Nodes are not part of this compose file — they run on hardware at the roadside.

```bash
docker build -f deploy/Dockerfile.edge -t thescanner-edge .
```

```bash
scanner-edge init --out node.yaml
```

Then on the node, with `/var/lib/thescanner` on a **persistent volume**:

```bash
docker run -d --restart unless-stopped -v /var/lib/thescanner:/var/lib/thescanner -v /etc/thescanner:/etc/thescanner:ro thescanner-edge
```

That volume holds the node's signing key and its store-and-forward queue.
Losing it loses undelivered reads and — worse — forks the node's evidence chain
on restart, making every subsequent read unverifiable.

Enrol the node's public key with the platform before it will be accepted:

```bash
scanner-edge enrol --config node.yaml
```

## Jetson

Swap the base image in `Dockerfile.edge` for an NVIDIA L4T image and install
`onnxruntime-gpu` from NVIDIA's index — it is not on PyPI for ARM. The rest is
unchanged; the ONNX artefact and the agent code are identical.

## Not yet done

Honest list, tracked in [`../docs/PLAN.md`](../docs/PLAN.md) Phase 6:

- No Helm chart yet — the national tier is designed but not packaged.
- Nothing here has been run against a live PostgreSQL with TimescaleDB; the
  schema is dialect-conditional and the test suite runs on SQLite.
- No load or chaos testing: the 72-hour offline tolerance and the throughput
  targets in the plan are unverified.
- OIDC and enforced MFA are not wired up. The local HMAC tokens are a fallback,
  not the intended production identity path.
