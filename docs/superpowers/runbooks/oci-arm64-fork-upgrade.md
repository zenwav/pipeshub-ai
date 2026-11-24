# Upgrading the OCI ARM64 server to this fork without losing data

Revised 2026-11-25. Target: the Ubuntu 26.04 ARM64 VPS (`oci-vps-2`) where PipesHub runs from
`/home/ubuntu/pipeshub` as Compose project **`pipeshub-ai`**, behind host Nginx, with a hand-edited
`docker-compose.yml` (MCP scopes and chat-off settings, Docling page limit, the crawl4ai profile).

Three rules that hold for every command below:

1. **Always pass `-p pipeshub-ai`** to `docker compose`. Without it Compose names the project after the folder
   (`pipeshub`), and `up` starts a second, empty PipesHub with new volumes beside the real one.
2. **Never run `down -v`**, and never delete `.env`. `-v` deletes the volumes; `.env` holds `SECRET_KEY`,
   without which saved API keys cannot be decrypted.
3. **Never replace `docker-compose.yml` wholesale.** Merge the new one with your edits (step 4).

## Step 1: Build an image from the v1.3.0 release, not from `main`

The fork's `main` is upstream's unreleased development branch. For the server, put the NVIDIA change on top of
the last release (`v1.3.0`, 22 Nov 2026). On your PC, in the fork clone (Git Bash):

```bash
cd /d/github-workspace/nginx-setup/nginx-pipeshub-research/pipeshub-ai
git status --short                    # the staged NVIDIA change, plus the untracked .codegraph/
rm -rf .codegraph                     # not part of the commit

# Commit the change on main (the workflow must be on the default branch to show "Run workflow").
git commit -S -m "feat(embeddings): native NVIDIA NIM embedding provider with container-wide pacing"
git push origin main

# A deploy branch: the release plus that one commit.
git fetch https://github.com/pipeshub-ai/pipeshub-ai.git refs/tags/v1.3.0:refs/tags/v1.3.0
git switch -c deploy/v1.3.0-nvidia v1.3.0
git cherry-pick -S main
git push -u origin deploy/v1.3.0-nvidia
git switch main
```

The change applies to `v1.3.0` without conflicts (checked on 25 Nov 2026). If `cherry-pick` stops on a
conflict, run `git cherry-pick --abort` and stop.

Then on GitHub:

1. Docker Hub: create a Personal Access Token with **Read & Write** scope.
2. Fork **Settings > Secrets and variables > Actions**: add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.
3. **Actions > Fork: Build and publish ARM64 slim image > Run workflow**. Choose
   **Use workflow from: `deploy/v1.3.0-nvidia`**, set **tag** to `v1.3.0-nvidia`, keep **push_sha_tag** on.
4. When it finishes, note the immutable tag in the run summary: `v1.3.0-nvidia-<7-char-sha>`. Use that one
   on the server, not the moving tag, so a later build never replaces what you run without you knowing.

The workflow builds on upstream's current `python-deps-slim` and `runtime` base images, as upstream's own
release build does. They are mutable tags; a much later rebuild may pick up newer dependencies.

## Step 2: Read-only checks on the server

```bash
cd /home/ubuntu/pipeshub
uname -m                                                   # aarch64
docker compose -p pipeshub-ai ps                           # the running services, all healthy
APP=$(docker compose -p pipeshub-ai ps -q pipeshub-ai)
docker inspect --format 'Image={{.Config.Image}}
Version={{index .Config.Labels "org.opencontainers.image.version"}}
ImageID={{.Image}}' "$APP"
grep -nE 'image: \$\{PIPESHUB_IMAGE|SANDBOX_DOCKER_IMAGE' docker-compose.yml
grep -E '^(COMPOSE_PROJECT_NAME|IMAGE_TAG|PIPESHUB_IMAGE|SANDBOX_DOCKER_IMAGE|COMPOSE_PROFILES|APP_PORT)=' .env
docker volume ls --filter label=com.docker.compose.project=pipeshub-ai
docker system df -v | sed -n '/VOLUME NAME/,/^$/p'         # volume sizes, for the backup
df -h /home                                                # free space, for the backup
```

If `Version` is newer than `1.3.0`, stop: moving back to v1.3.0 may not read data a newer version wrote.

## Step 3: Back up before changing anything

The backup needs about as much free space as the volumes use. The stack is stopped while it runs, so the
databases are copied in a consistent state.

```bash
cd /home/ubuntu/pipeshub
TS=$(date +%Y%m%d-%H%M%S)
B=/home/ubuntu/pipeshub-backups/$TS
mkdir -p "$B" && chmod 700 /home/ubuntu/pipeshub-backups "$B"
cp -p .env docker-compose.yml "$B/"

# Keep the app and sandbox images you run now under local names, for rollback (step 8).
APP=$(docker compose -p pipeshub-ai ps -q pipeshub-ai)
docker inspect --format '{{.Image}}' "$APP" > "$B/app-image-id.txt"
docker tag "$(cat "$B/app-image-id.txt")" pipeshub-rollback/pipeshub-ai:pre-nvidia

SBX=$(docker compose -p pipeshub-ai ps -a -q sandbox-image | head -n 1)
if [ -n "$SBX" ]; then
  docker inspect --format '{{.Image}}' "$SBX" > "$B/sandbox-image-id.txt"
  docker tag "$(cat "$B/sandbox-image-id.txt")" pipeshub-rollback/pipeshub-sandbox:pre-nvidia
elif docker image inspect pipeshubai/pipeshub-sandbox:slim >/dev/null 2>&1; then
  docker image inspect --format '{{.Id}}' pipeshubai/pipeshub-sandbox:slim > "$B/sandbox-image-id.txt"
  docker tag pipeshubai/pipeshub-sandbox:slim pipeshub-rollback/pipeshub-sandbox:pre-nvidia
else
  docker pull pipeshubai/pipeshub-sandbox:slim
  docker image inspect --format '{{.Id}}' pipeshubai/pipeshub-sandbox:slim > "$B/sandbox-image-id.txt"
  docker tag pipeshubai/pipeshub-sandbox:slim pipeshub-rollback/pipeshub-sandbox:pre-nvidia
fi

# Stop everything (all profiles). Volumes are kept: no -v.
docker compose -p pipeshub-ai --profile '*' down --remove-orphans

# One archive per volume.
for v in $(docker volume ls -q --filter label=com.docker.compose.project=pipeshub-ai); do
  echo "backing up $v"
  docker run --rm -v "$v":/data:ro -v "$B":/backup alpine:3.22 tar -C /data -czf "/backup/$v.tar.gz" .
done
ls -lh "$B"
```

Check that every volume from step 2 has a `.tar.gz` of plausible size before going on.

## Step 4: Merge the new compose file with your edits

Fetch the deploy branch's compose file next to yours, without replacing it:

```bash
cd /home/ubuntu/pipeshub
curl -fsSL -o docker-compose.v1.3.0-nvidia.yml \
  https://raw.githubusercontent.com/<your-github-user>/pipeshub-ai/deploy/v1.3.0-nvidia/deployment/docker-compose/docker-compose.yml
diff -u docker-compose.yml docker-compose.v1.3.0-nvidia.yml | less
```

In the diff, lines starting with `-` are in your file only: those are your edits (and lines upstream removed).
Copy your edits into `docker-compose.v1.3.0-nvidia.yml` with `nano`, at the same places. At least check:

- your `MCP_SCOPES` value, and the `- PIPESHUB_MCP_CHAT=${PIPESHUB_MCP_CHAT:-}` line;
- any `ports:` change you made (the app published on host port 9435);
- `PDF_DOCLING_MAX_PAGES` and any other setting you added by hand.

The new file already has the two NVIDIA lines (`NVIDIA_NIM_RPM`, `NVIDIA_NIM_INTERACTIVE_RESERVE_RPM`).
Then validate it, still without touching the live file:

```bash
docker compose -p pipeshub-ai -f docker-compose.v1.3.0-nvidia.yml config --quiet && echo "compose OK"
```

Only when that passes:

```bash
cp -p docker-compose.v1.3.0-nvidia.yml docker-compose.yml
```

Your original is in `$B/docker-compose.yml`.

## Step 5: Point the app at the fork image

Edit `.env` with `nano` (do not touch `SECRET_KEY` or any password). Set or add:

```
PIPESHUB_IMAGE=<your-dockerhub-user>/pipeshub-ai
IMAGE_TAG=v1.3.0-nvidia-<7-char-sha>
SANDBOX_DOCKER_IMAGE=pipeshubai/pipeshub-sandbox:1.3.0-slim
```

`SANDBOX_DOCKER_IMAGE` is needed because the default sandbox image name is built from `IMAGE_TAG`, and
upstream has no `v1.3.0-nvidia-...` sandbox. `pipeshubai/pipeshub-sandbox:1.3.0-slim` is upstream's sandbox
for the same release (published with arm64). Check what Compose will run:

```bash
docker compose -p pipeshub-ai config | grep -nE 'image:|SANDBOX_DOCKER_IMAGE'
```

The `pipeshub-ai` service (and `docker-socket-proxy`, if you use the sandbox profile) must show your image and
tag; every other service keeps its usual image.

Optional, in `.env`: `NVIDIA_NIM_RPM` (requests a minute for the key, shared by every PipesHub service in the
container; default 36 on NVIDIA Build) and `NVIDIA_NIM_INTERACTIVE_RESERVE_RPM` (how many of those indexing
leaves for searches; default 6). The defaults suit a free key.

## Step 6: Start and check

Start the same services you ran before (add `--profile` options you normally use; you start the web readers
with the `fetch-crawl4ai` profile):

```bash
cd /home/ubuntu/pipeshub
docker compose -p pipeshub-ai pull pipeshub-ai sandbox-image
docker compose -p pipeshub-ai up -d
docker compose -p pipeshub-ai --profile fetch-crawl4ai up -d web-readers-gateway crawl4ai
watch -n 10 'docker compose -p pipeshub-ai ps'      # wait until pipeshub-ai shows (healthy); Ctrl+C
docker compose -p pipeshub-ai logs --since 10m pipeshub-ai | grep -iE "error|nvidia" | tail -40
```

Pulling `sandbox-image` explicitly before `up -d` matters because `sandbox-image` in `docker-compose.yml` also
declares a `build:` section; if `pipeshubai/pipeshub-sandbox:1.3.0-slim` is not pulled first, Compose tries to
build `../../deployment/sandbox` locally, which does not exist on the server.

The Python services take a few minutes; "service unavailable" in the meantime is expected. Then open the site
through Nginx, sign in, and check that your chat model and saved keys are still there.

## Step 7: Switch embeddings to NVIDIA NIM

1. **Workspace settings > AI models > For embedding > NVIDIA NIM**.
2. API key: your `nvapi-...` key. Model: `nvidia/nemotron-3.5-embed-8b`. Endpoint: leave blank.
   Dimensions: leave blank for the native 4096.
3. **Multimodal is off by default.** Turn it on to embed slide and figure images as well; each batch of up
   to 4 images is one more request against the per-minute limit.
4. Save. The check sends one text request (and one image request with Multimodal on).
5. Your collection still holds vectors from the previous embedding model, of a different size and meaning.
   Clear the old embeddings and re-index as PipesHub asks when you change the model (earlier, for the Gemini
   switch, that was under **Labs**). I could not confirm the exact menu names for this version.

While indexing runs, NVIDIA requests stay at about 30 a minute, and searches have 6 a minute of their own.
A record that hits NVIDIA's limit anyway is put back in the queue and retried, not failed.

## Step 8: Roll back

Use this if the new version misbehaves.

**Image only** (the data was not changed, or you accept what the new version wrote): in `.env`, set

```
PIPESHUB_IMAGE=pipeshub-rollback/pipeshub-ai
IMAGE_TAG=pre-nvidia
SANDBOX_DOCKER_IMAGE=pipeshub-rollback/pipeshub-sandbox:pre-nvidia
```

(`SANDBOX_DOCKER_IMAGE` must also be set because an empty `SANDBOX_DOCKER_IMAGE` expands from `IMAGE_TAG` to a
nonexistent `pipeshubai/pipeshub-sandbox:pre-nvidia`.) Restore your old compose file, then start all services
(including the `fetch-crawl4ai` web readers) without pulling:

```bash
cd /home/ubuntu/pipeshub
cp -p "$B/docker-compose.yml" docker-compose.yml
docker compose -p pipeshub-ai --profile '*' down --remove-orphans
docker compose -p pipeshub-ai up -d --pull never
docker compose -p pipeshub-ai --profile fetch-crawl4ai up -d --pull never web-readers-gateway crawl4ai
```

(`$B` is the backup folder from step 3; if your shell was closed, set it again, for example
`B=/home/ubuntu/pipeshub-backups/<timestamp>`.)

**Image and data** (the new version changed the data and the old one will not start on it): stop the stack,
restore `.env`, `docker-compose.yml`, and every volume from its archive:

```bash
cd /home/ubuntu/pipeshub
cp -p "$B/.env" .env && chmod 600 .env
cp -p "$B/docker-compose.yml" docker-compose.yml
docker compose -p pipeshub-ai --profile '*' down --remove-orphans
for f in "$B"/*.tar.gz; do
  [ -e "$f" ] || { echo "no archives found in $B"; break; }
  v=$(basename "$f" .tar.gz)
  echo "restoring $v"
  docker run --rm -v "$v":/data -v "$B":/backup alpine:3.22 \
    sh -c 'find /data -mindepth 1 -delete && tar -C /data -xzf "/backup/$1.tar.gz"' _ "$v"
done
```

Restoring the original `.env` resets `SANDBOX_DOCKER_IMAGE=` to empty and `IMAGE_TAG=slim`. Edit `.env` to set
the three `pipeshub-rollback` lines above (`PIPESHUB_IMAGE`, `IMAGE_TAG`, and `SANDBOX_DOCKER_IMAGE`), so the
exact app and sandbox images saved in step 3 are used rather than whatever `:slim` points to today, then start:

```bash
docker compose -p pipeshub-ai up -d --pull never
docker compose -p pipeshub-ai --profile fetch-crawl4ai up -d --pull never web-readers-gateway crawl4ai
```
