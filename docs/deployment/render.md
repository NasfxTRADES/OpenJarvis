# Deploy the phone app on Render

The repository's `render.yaml` is a Render Blueprint for the OpenJarvis phone
app. It builds `deploy/docker/Dockerfile` and runs the same `jarvis mobile`
server you would run on your own computer, so the iPhone app is reachable over
HTTPS while your computer is off.

## Create the service

1. In the Render Dashboard click **+ New → Blueprint**. Render opens a list of
   your repositories: press **Connect** next to your OpenJarvis fork.
   If the fork is not in that list, the Render GitHub App cannot see it — grant
   access at <https://github.com/apps/render/installations/new>. Repositories
   are not found by searching the dashboard; only by connecting them here.
2. On the form that follows, set **Branch** to the branch that contains the
   phone app (`main` once this work is merged, otherwise the feature branch),
   leave **Blueprint Path** as `render.yaml`, and press **Deploy Blueprint**.

   The branch field matters: the Dashboard reads `render.yaml` from the default
   branch unless you change it, and the default branch may still describe the
   desktop server that asks for an OpenAI key.
3. No secret is required at creation — both provider keys are optional. Wait
   for the build to finish and `/health` to report healthy; the first build
   compiles the Rust core and takes several minutes.
4. Copy the generated `OPENJARVIS_API_KEY` from the service's **Environment**
   tab and open `https://<service-name>.onrender.com/m` on the phone, then
   paste that key once; the phone remembers it. Use Safari's
   **Share → Add to Home Screen** to install it.

Alternatively set `OPENROUTER_API_KEY` in the dashboard instead of
`GEMINI_API_KEY` to use OpenRouter's free models; `jarvis mobile` picks
whichever provider key is present. Leave the other one empty.

The Blueprint pins both the `PORT` environment variable and the server's
`--port` argument to `10000`. Render's Docker command does not expand `$PORT`,
so if you customize the service port, update both values in `render.yaml`.

Render's **Docker Command** replaces both the image's `CMD` *and* its
`ENTRYPOINT`, so the Blueprint spells out the executable:
`/usr/local/bin/jarvis mobile --host 0.0.0.0 --port 10000`. If you shorten it to
`mobile --port 10000`, Render tries to run a program named `mobile`, and the
deploy ends with the uninformative `Exited with status 128` and no app logs.

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
