# Handoff: Convograph — realtime conversation → knowledge graph → graphic recording

## Overview
Convograph listens to a live conversation (up to 4 people), transcribes it with speaker IDs, cuts the stream into **episodes**, extracts a **knowledge graph** per episode (facts + relations, with validity tracking), and continuously repaints a **graphic recording** (Excalidraw sketch layer + Flux painterly render). This package covers four desktop screens: Live session, Speaker identification, Episode diff (fact history), and Settings.

## About the Design Files
`Convograph v2.dc.html` (and the earlier serif variant `Convograph.dc.html`) are **design references built in HTML** — prototypes showing intended look and behavior, not production code. Recreate them in the target codebase's environment (React/Next, Svelte, etc.) using its established patterns. If no codebase exists yet, a React + TypeScript app with a canvas/SVG layer for the graph is the natural fit. The HTML uses a proprietary streaming template runtime (`support.js`, `<x-dc>`, `{{ }}` holes) — ignore that plumbing; read the inline styles and structure.

Primary reference: **`Convograph v2.dc.html`** (modern sans-serif). `Convograph.dc.html` is the earlier editorial/serif direction, kept for comparison only.

## Fidelity
**High-fidelity.** Colors, type, spacing, radii and copy are final for v2. Recreate pixel-close at 1440×900; the layout should also reflow down to ~1200px (transcript rail stays 300px, graph 420px, recording takes the rest).

## Screens / Views

### 1a — Live session
- **Purpose**: all three modules running; user watches transcript, graph, and recording grow.
- **Frame**: 1440×900, background `#f5f5f7`. Grid rows `56px | 1fr | 64px`.
- **Top bar (56px)**: flex, gap 22px, padding 0 24px. Brand "Convograph" 17px/600, letter-spacing −0.02em. 1px×20px vertical divider `rgba(0,0,0,.08)`. Session title 13px. Live indicator: 8px blue dot + pulsing ring (scale 1→1.6, opacity .9→.25, 1.8s ease-in-out infinite) + "Listening · 12:48:06" 12px `#5b5b63`, tabular numerals. Right cluster: "Speakers" label + 4 overlapping 22px monogram circles (margin-left −6px), buttons Pause (secondary), Export (secondary), End session (primary), Settings icon button (34px).
- **Body**: grid `300px 420px 1fr`, gap 12px, padding 12px. Each panel: white card, radius 14px, shadow `0 1px 2px rgba(0,0,0,.04), 0 0 0 1px rgba(0,0,0,.05)`, overflow hidden, flex column.
  - **Transcript panel**: header row padding 16px 18px 10px — "Transcript" 11.5px/500 `#5b5b63` + word count 11px. Turns: grid `22px 1fr`, gap 8px, 12.5px/1.55. Monogram circle 22px, 1px border in speaker color, initial 11px/500. Name 12.5px/500, timestamp 11px `#76767f`. Episode boundary: hairline–label–hairline, 10px uppercase 0.1em tracking; current episode uses accent-300 lines and accent-700 text. Older episode turns at opacity .55–.75. Live turn: monogram gets `box-shadow 0 0 0 3px #eaf1ff`, timestamp replaced by "speaking" in accent-700, text ends with a 1.5×13px blinking caret (1s steps). Footer: 6 audio bars 2px wide (heights 5–14px) + "Studio mic · Whisper-large · 240 ms" 11px.
  - **Knowledge graph panel**: header "Knowledge graph" + "14 facts · 19 relations". Body: full-size SVG, nodes on a sphere (Fibonacci distribution), rotating about Y at 0.18 rad/s, perspective scale `1 + z·0.35`, depth opacity `0.3 + (z+1)·0.35`, edges 0.8px `#1d1d1f` at opacity `0.1 + (z+1)·0.16`. Node types: confirmed = white fill, 1px `#5b5b63` stroke; added this episode = solid `#0a66ff` fill, radius pulses ±12% (sin, ~2.8s); revised = 1.6px `#0a66ff` stroke; invalidated = dashed `2 2` stroke `#9d9da5`, label line-through `#9d9da5`. Labels Geist 9.5–13px, offset `r+5px` right. Nodes re-sorted by depth each frame. Bottom-left legend (3 rows, 11px). Bottom-right: −/+ icon buttons 28px, ghost link "Compare episodes →". Footer status: pulsing 6px dot + "Extracting from episode 8 — *Bee hives* marked invalid, *Soil test* added".
  - **Graphic recording panel** (hero): header "Graphic recording", meta "Painterly · Flux 1.1 · 1536×1024", segmented control Latest / Frames / Sketch layer (Latest selected). Body padding 0 22px 12px: image plate radius 14px, contains (a) the Flux render (image), (b) an SVG sketch layer on top — 1.3px `#2d2b2b` strokes, hand-drawn ellipses around labels, arrows, dashed ellipse + ✕ over invalidated fact, (c) HTML labels 13px/500 positioned by %, invalidated label line-through at 55% opacity, (d) a slow "brushing-in" light sweep (28% wide gradient, translateX −100%→320%, 5s linear infinite), (e) bottom-left status chip: blurred white 82% bg, 1px border, radius 8px, 11px — pulsing dot, "Rendering episode 8 · pass 2 of 3", 90px hairline progress (62% accent), "62%". Footer strip: "Frames" label 11px (64px wide), 7 thumbnails 54×36px (opacity ramps .5→1, latest has 1px accent border), dashed placeholder "8 …", "Play" label + 28px play icon button.
- **Episode track (64px)**: grid `300px 1fr 200px`, padding 0 24px, gap 20px. Left: "**8** episodes · 7 graphed · 6 rendered" 11.5px. Middle: 8 columns, each 3 bars (2px tall) = transcribed / graphed / rendered; done bars `#5b5b63`, in-progress `#0a66ff` (partial at 55% opacity), not started dashed `#c4c4ca`; label "n · hh:mm" 10px tabular, live one in accent-700. Right: legend + "Episode ≈ 5 min or a topic shift" 10.5px.

### 1b — Speaker identification (first minutes)
Same shell as 1a. Differences:
- Title italic placeholder "Untitled session — name it when you like" `#76767f`. Timer "Listening · 00:01:52". Tag (accent pill) "3 of 4 voices identified".
- Transcript header hint "Click a name to rename". Speaker names have a 1px dotted bottom border (`#c4c4ca`) = renameable. **Inline rename state**: name replaced by a 110×24px input (accent border, 12.5px) + "↵ to save" hint 10px; below the turn a card (elevation md, white, 1px border, radius 8px, padding 10px 12px) with "Voice print · Speaker 3" kicker, a waveform SVG (1.2px strokes), and "3 turns · 41 s spoken" / "Applies to all past turns". Unidentified speaker: dashed monogram circle with a number, italic grey "Speaker 4".
- Graph: 4 nodes, radius factor 0.26, "4 facts · 3 relations"; footer "Episode 1 closes at ~05:00 or on a topic shift".
- Recording: empty state centered — 72px spinner (two concentric circles, outer has dash `40 150` rotating 3s linear), heading 22px/600 "The first plate arrives with episode 1", 13px body, "First render in about 3 min". Frames strip shows only dashed "1 …".
- Track: episode 1 live (transcribed bar 38% filled), 2–4 dashed at decreasing opacity.

### 1c — Compare episodes (fact history)
- Frame rows `56px | 1fr`. Top bar adds breadcrumb "/ Compare episodes", status "Still listening · episode 9", buttons "Export ledger" (secondary), "Back to session" (primary).
- **Scrubber** (padding 26px 40px 18px, bottom hairline): h2 24px/600 "Episode 5 → Episode 8" (arrow `#9d9da5`); right summary "+3 added · 3 revised · 1 confirmed · ~~1~~ invalidated · 7 unchanged" 12px tabular. Track: 1px `#e2e2e6` line, selected span 3px accent between handles A (ep 5, hollow 15px circle, 1.5px accent border, white fill) and B (ep 8, filled accent). Episode dots 7px: past `#9d9da5`, within range accent, live dashed accent. Labels 10.5px tabular; handles labeled "5 · 12:23 A" / "8 · 12:41 B".
- **Ledger** (left, padding 22px 40px): segmented "Changes only / All facts", "Sorted by episode of change" 12px. Table 13px: columns Fact (26%) · At episode 5 (25%) · At episode 8 (25%) · Change · By (right-aligned 20px monogram). Old values grey `#76767f` with line-through when revised; "—" `#9d9da5` when absent. Change tags (pill, 11px/500): Added = accent tint (`#eaf1ff` / `#0a3480`), Revised = neutral tint (`#fafafa` / `#3d3d44`), Confirmed = outline accent, Invalidated = neutral tint with line-through. Footnote 11.5px `#76767f`: rows link to the causing transcript turn; invalidated facts stay as dashed nodes.
- **Changed region** (right, 480px, left hairline): "Changed region" / "Unchanged facts faded"; same 3D graph with unchanged nodes at 35% opacity, radius factor 0.34; footer legend Added / Revised / Invalidated.

### 1d — Settings sheet
- Opens over the live session: session dimmed to 35% opacity + 1.5px blur, scrim `rgba(29,29,31,.22)`.
- Sheet: right-anchored 560px, white, radius 18px 0 0 18px, shadow lg. Rows `auto | 1fr (scroll) | auto`. Header padding 26px 32px 14px: "Session settings" 24px/600 + ghost "Close". Body padding 8px 32px, scrollable; each section grid `150px 1fr`, gap 20px, padding 18px 0, hairline between sections. Section label 11.5px/500 in accent + 11.5px grey description.
  - **Listening**: Microphone (select-style input "Studio Condenser USB"), Speakers expected segmented 2 / 3 / **4** / Detect, Transcription model ("Whisper large-v3 · English + German").
  - **Episodes**: radios — **Every 5 minutes, or sooner on a topic shift** / Only on a topic shift / Fixed length; Graph model ("Convograph KG · v2 (entities, relations, validity)"); checkbox **Keep invalidated facts as dashed nodes** (checked).
  - **Recording**: Style — 3 swatch cards (Sketchnote, Diagrammatic, **Painterly ✓** selected with accent border + inset ring), Image backend ("Flux 1.1 pro · 1536×1024 · 3 passes"), Repaint segmented **Every episode** / Every 2 / On demand, Palette prompt textarea (64px) "Warm ochre and sage, loose oil strokes, archival paper ground, labels lettered by hand."
  - Footer: "Changes apply from the next episode" 11.5px + Reset (secondary) / Save (primary).

## Interactions & Behavior
- **Live indicator**: pulse ring 1.8s; caret blink 1s steps(1); graph node pulse; recording status dot 1.4s; sweep 5s linear. All infinite while state = listening; freeze when paused.
- **Transcript**: auto-follows the newest turn; older episodes fade (opacity .55→1 across last ~5 turns). Episode boundaries inserted when the episodizer cuts (5 min or topic shift). Clicking a speaker name → inline input; Enter saves and renames all past/future turns of that voice; Esc cancels. Voice-print card appears while editing.
- **Graph**: continuous slow rotation (0.18 rad/s), drag to rotate (optional), wheel/± to zoom. New nodes fade in at full accent fill and pulse until the next episode is processed, then settle to "confirmed" styling. Invalidated nodes switch to dashed and stay. Hovering a node highlights its edges and (next step) shows source turns. "Compare episodes →" navigates to 1c.
- **Recording**: shows latest render; "Frames" toggles a filmstrip/timeline view; "Sketch layer" shows the Excalidraw layer alone. Status chip shows render pass progress. Play button plays frames as a film.
- **Episode track**: each episode has 3 stages; stage bar fills when the stage completes; live episode's transcribed bar fills proportionally to time elapsed.
- **Compare (1c)**: two draggable handles A/B snapping to episodes; ledger recomputes diff between graph states at A and B; "Changes only / All facts" filter; row click jumps to the transcript turn that caused the change (highlight it). Live episode keeps appending as a dashed dot.
- **Settings**: slide-in from right (≈240ms ease-out), Esc/Close/scrim click dismisses; Save applies from next episode; Reset restores defaults.
- **Pause**: stops audio capture; indicator dot stops pulsing, label "Paused"; graph rotation may continue.
- Focus: 2px accent outline, offset 2px, on all interactive elements.

## State Management
- `session`: { id, title, startedAt, state: 'idle'|'listening'|'paused'|'ended', speakersExpected, settings }
- `speakers[]`: { id, label, initial, colorStep (accent-500 / accent-800 / neutral-500 / neutral-900), voicePrint, turnsCount, spokenSeconds, named: bool }
- `turns[]`: { id, speakerId, tStart, text, isFinal, episodeId }
- `episodes[]`: { id, index, tStart, tEnd?, stages: { transcribed, graphed, rendered } each 'pending'|'running'(progress)|'done' }
- `graph`: nodes { id, label, size, status: 'confirmed'|'added'|'revised'|'invalidated', addedInEpisode, changedInEpisode, sourceTurnIds[] }, edges { a, b }
- `graphSnapshots[episodeId]` for diffing; `diff(A,B)` → rows { fact, valueAtA, valueAtB, change, episode, bySpeakerId }
- `renders[]`: { episodeId, imageUrl, sketchSvg, labels[], pass, totalPasses, progress }
- Streams: ASR partial/final turns (WebSocket), episodizer events, KG updates (node add/revise/invalidate), render progress + completion.

## Design Tokens (v2)
- **Fonts**: Geist (Google Fonts, 400/500/600), fallback -apple-system, system-ui, sans-serif. Body 15px/1.55, letter-spacing −0.005em; headings 600, letter-spacing −0.025em. Tabular numerals wherever figures are compared.
- **Ground/text**: bg `#f5f5f7`, surface/card `#ffffff`, text `#1d1d1f`, divider `rgba(0,0,0,.08)`, page canvas behind frames `#e9e9ec`.
- **Accent (blue)**: 100 `#eaf1ff`, 200 `#d3e2ff`, 300 `#a9c6ff`, 400 `#6f9fff`, 500 `#2f78ff`, base `#0a66ff`, 600 `#0a5be6`, 700 `#0846b4`, 800 `#0a3480`, 900 `#0b234f`.
- **Neutral**: 100 `#fafafa`, 200 `#f0f0f2`, 300 `#e2e2e6`, 400 `#c4c4ca`, 500 `#9d9da5`, 600 `#76767f`, 700 `#5b5b63`, 800 `#3d3d44`, 900 `#1d1d1f`.
- **Speaker colors**: M accent-500 / T accent-800 / I neutral-500 / K neutral-900 (border + initial; white fill).
- **Radii**: sm 4, md 8, lg 14, cards 14, frame 18, controls pill 999.
- **Shadows**: sm `0 1px 2px rgba(0,0,0,.06)`, md `0 4px 14px rgba(0,0,0,.08)`, lg `0 20px 50px rgba(0,0,0,.14)`, card ring `0 1px 2px rgba(0,0,0,.04), 0 0 0 1px rgba(0,0,0,.05)`.
- **Spacing**: 4 / 8 / 12 / 16 / 18 / 22 / 24 / 32 / 40 px as used above; panel gap 12.
- **Controls**: Button pill, 13px/500, 34px tall in the top bar; primary filled accent (hover 600, active 700) white text; secondary white with `rgba(0,0,0,.1)` border; ghost accent text. Segmented: track `#f0f0f2` pill, 2px padding, selected option white with `0 1px 3px rgba(0,0,0,.1)`. Input 36px, radius 10, `rgba(0,0,0,.1)` border, accent on focus. Tag pill 11px/500, 3px 9px.
- **Painterly placeholder** (until real renders exist): radial gradients `#c9a56a`, `#8d9a7a`, `#e3d3b6`, `#7e6a4d` over `#efe6d7→#d9cdb8`, blur 2px, slow drift 24s.

## Assets
- Icons: Lucide (pause, download, settings, minus, plus, play, chevron-down) as inline SVG, 1.6px stroke.
- Graphic recording images: placeholders (`<image-slot>` drop targets) — replace with Flux output. Sketch layer is inline SVG meant to be generated from Excalidraw JSON.
- No brand assets; the wordmark is plain type.

## Files
- `Convograph v2.dc.html` — primary reference (sans-serif / modern), all four screens side by side; canvas layout with ids `1a`–`1d`.
- `Convograph.dc.html` — earlier serif/editorial direction (reference only).
- `image-slot.js` — placeholder image component used by the prototypes (not needed in production).
- `_ds/` — design-system stylesheet the prototypes load; v2 overrides its tokens in the page head. Only the tokens listed above matter for implementation.
