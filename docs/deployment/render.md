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
3. No secret is required at creation — every provider key is optional. Wait
   for the build to finish and `/health` to report healthy; the first build
   compiles the Rust core and takes several minutes.
4. Copy the generated `OPENJARVIS_API_KEY` from the service's **Environment**
   tab and open `https://<service-name>.onrender.com/m` on the phone, then
   paste that key once; the phone remembers it. Use Safari's
   **Share → Add to Home Screen** to install it.

## Automatic free-provider failover

You do not have to pick a provider. Fill in as many keys as you like in the
service's **Environment** tab and `jarvis mobile` chains them, in this order:

| # | Provider | Environment variable | Free tier |
|---|----------|---------------------|-----------|
| 1 | Groq | `GROQ_API_KEY` | Yes, per model — `console.groq.com/keys` |
| 2 | Cerebras | `CEREBRAS_API_KEY` | Yes — `cloud.cerebras.ai` |
| 3 | Google Gemini | `GEMINI_API_KEY` | Yes, Flash models — `aistudio.google.com/apikey` |
| 4 | Mistral | `MISTRAL_API_KEY` | Yes, experimentation tier — `console.mistral.ai/api-keys` |
| 5 | OpenRouter | `OPENROUTER_API_KEY` | Yes, zero-priced models — `openrouter.ai/keys` |
| 6 | Grok (xAI) | `XAI_API_KEY` | Free credits on signup — `console.x.ai` |
| 7 | ChatGPT (OpenAI) | `OPENAI_API_KEY` | No, billed per token |
| 8 | Claude (Anthropic) | `ANTHROPIC_API_KEY` | No, billed per token |

When a provider answers with a rate-limit or quota error (HTTP 429 / 402 /
`RESOURCE_EXHAUSTED`), Jarvis marks it as resting and immediately retries the
same request with the next provider, so a chat reply still arrives. Quota
errors rest a provider for an hour, other failures for a minute; when the
cooldown ends it returns to its place in the chain. Several models per
provider join the chain too — Groq's free allowance, for example, is counted
per model, so its three free chat models each add headroom.

Reorder or restrict the chain with `OPENJARVIS_FREE_CHAIN`, for example
`OPENJARVIS_FREE_CHAIN="gemini,groq,openrouter"` to keep everything else out.
Any provider you leave out of the list is still tried afterwards, so the
quickest way to *disable* a provider is to leave its key unset.

The phone's **Settings** tab shows the live chain, which provider is answering,
and which ones are resting; **Rebuild free chain** re-runs discovery after you
change keys. Choosing a single provider in Settings (`Connect Gemini`, say)
overrides the chain until you rebuild it.

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
