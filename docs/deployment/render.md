# Deploy the phone app on Render

The repository's `render.yaml` is a Render Blueprint for the OpenJarvis phone
app. It builds `deploy/docker/Dockerfile` and runs the same `jarvis mobile`
server you would run on your own computer, so the iPhone app is reachable over
HTTPS while your computer is off.

## Create the service

1. Open Render's **New Blueprint Instance** flow and select your OpenJarvis
   fork, on the branch that contains the phone app.
2. Provide `GEMINI_API_KEY` when Render asks for the Blueprint secret — a free
   key from <https://aistudio.google.com/apikey> is enough.
3. Apply the Blueprint and wait for `/health` to become healthy. The first build
   compiles the Rust core and takes several minutes.
4. Copy the generated `OPENJARVIS_API_KEY` from the dashboard and open
   `https://<service-name>.onrender.com/m` on the phone, then paste that key
   once; the phone remembers it. Use Safari's **Share → Add to Home Screen** to
   install it.

Alternatively set `OPENROUTER_API_KEY` in the dashboard instead of
`GEMINI_API_KEY` to use OpenRouter's free models; `jarvis mobile` picks
whichever provider key is present. Leave the other one empty.

The Blueprint pins both the `PORT` environment variable and the server's
`--port` argument to `10000`. Render's Docker command does not expand `$PORT`,
so if you customize the service port, update both values in `render.yaml`.

## What the free instance does and does not keep

!!! warning "Free instances do not preserve OpenJarvis data"
    Render's free web-service filesystem is ephemeral. Everything written to
    `/home/openjarvis/.openjarvis` — scheduled tasks, the saved model choice,
    transcripts and learned state — is lost whenever the service restarts,
    spins down, or redeploys. Provider keys survive because they are
    environment variables, not files.

Free instances also spin down after a period without inbound traffic, so the
first request after an idle spell waits for a cold start. For durable state,
upgrade to a paid instance and attach a persistent disk at
`/home/openjarvis/.openjarvis`.

Voice input is not available in this image because the speech extra is not
installed; chat and scheduled tasks work normally.
