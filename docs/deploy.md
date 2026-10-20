# Deploying

The browser only allows camera access on **HTTPS** (or `localhost`), so put TLS in front.

```bash
docker compose up --build            # API + web client on :8000
```
Mount your trained model into `./models/signs.npz` (see `docker-compose.yml`).

## HTTPS with Caddy (automatic certificates, WebSocket works out of the box)
```
signs.example.com {
    reverse_proxy aperture:8000
}
```

## Checklist
- [ ] `APERTURE_CORS_ORIGINS` lists only your real site origin(s).
- [ ] `APERTURE_ENABLE_COLLECT` is `0` on any public server. Recording writes files and has no
      login; enable it only on a private/dev instance, or behind authentication.
- [ ] `models/` is a persistent volume (train offline, copy `signs.npz` in, press "Reload model").
- [ ] Rate-limit `/ws/recognize` and `/pipeline` at the proxy if the site is public.
- [ ] Recorded landmark data is personal data: get consent, store it privately, honour deletion requests.
