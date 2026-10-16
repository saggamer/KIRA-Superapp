# KIRA Superapp website

A responsive, dependency-free product website covering chat and voice, research, artifact creation, coding, tools and workflows, and local memory. KIRA Live 1 has a dedicated section explaining text chat, native live voice, interruption, context, and the connection to KIRA’s guarded agent runtime. Model cards distinguish Live 1, Orchestrator V1, and Orchestrator v1.5, which is under development with training paused.

The visual experience combines original ChatGPT-generated white-hole artwork with WebGL plasma flow, cursor parallax, floating product notes, light editorial sections, and an animated voice sculpture. The final section shrinks the white hole into a point and then darkness. Motion can be paused, reduced-motion preferences are respected, and the original image renders when WebGL is unavailable. Branding is plain text; no invented logo is used.

The six capability tabs support keyboard navigation. The interactive workflow preview has research, presentation, and project examples, with interruption/resume, an explicit example approval, result verification, replay, and reset. This is a local illustrative simulation: it does not access the microphone, call a model, change files, or execute tools.

Artwork: `assets/white-hole-cinematic-v3.png`. The exact image generation brief is recorded in `assets/white-hole-generation-prompt.txt`.

## Local preview

```sh
python3 -m http.server 4173 --bind 127.0.0.1
```

Open http://127.0.0.1:4173.

## Vercel

Published production URL: https://kira-superapp.vercel.app/

The `kira-superapp` Vercel project is linked from this directory. Use `website` as the project root, framework preset **Other**, no build command, and output directory `.`. All public files are contained here; no app runtime, model weights, or private files are required. `.vercelignore` excludes environment files, deployment metadata, project notes, and unused artwork.

For future updates, deploy only when requested, from this directory:

```sh
npx vercel deploy --prod --scope saggamer-s-projects
```

Social links: Instagram [SagGuyTalks](https://www.instagram.com/SagGuyTalks/) and X [@saggamer2241](https://x.com/saggamer2241).
