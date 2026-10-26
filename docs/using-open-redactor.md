# Using Open Redactor

A short guide to running it well, what stays private, and what it cannot do yet.

## The 30 second loop

Preview first, read the coverage result line, then run the full clip. See the README demo for the four commands. Preview costs 3 seconds of rendering and saves a bad full run.

## The local page

Run `open-redactor --ui` and open http://127.0.0.1:8765. Drop an MP4, pick a preset, submit. The server binds only to your own machine, writes output to a temp folder it prints, and runs the same CLI underneath. Nothing is hosted and no file leaves the machine from the page itself.

## Where SAM runs. Three backends

SAM 3.1 is an open weights model, so Open Redactor does not hard assume one home for it.

**API backend, the default.** `--backend api` posts your clip to the Meta Model API at api.meta.ai with your key from an environment variable. Best quality today with no GPU and no downloads. Your video does leave the machine, so do not use it for footage that must stay local.

**Hosted backend.** `--backend hosted --endpoint https://your-host.example/v1/responses` sends the same Responses API request shape to an endpoint you control, or set OPEN_REDACTOR_ENDPOINT once. Use this for a team server, a cloud GPU box, or a third party host that speaks the same shape. The API key still comes from the named environment variable and is sent only to that endpoint.

**Local backend.** `--backend local` runs Grounding SAM on your own hardware, so private footage never leaves the device. Grounding DINO finds the phrase, SAM 2 cuts the mask, and a small IoU tracker keeps identities stable across frames. Install the heavy stack once with pip install "open-redactor[local]" and run with --provider grounding-sam, which is the default for the local backend. It uses a GPU when one is present and falls back to CPU slowly. Without the optional stack it returns zero matches with a clear install message and never pretends it detected something. A native SAM video propagator and larger model options are the next local upgrades.

For a second local path, `--provider owlv2` runs Google's OWLv2 base ensemble detector on your own hardware from the same local extra. It accepts open vocabulary phrases and returns boxes rather than pixel masks, then links those boxes across frames with the shared IoU tracker. The default score threshold is 0.05, chosen for recall because a missed sensitive object costs more than an extra covered patch. On the public trail group scene that threshold returns 17 person boxes against SAM 3.1's 16 people, while 0.1 returns only 4. The summary records the box mask source, so an OWLv2 run never presents itself as a pixel mask SAM run. Grounding SAM remains the local choice when mask shape matters.

For the simplest job, faces only, `--provider yunet` is the basic tier: OpenCV's YuNet detector runs locally in milliseconds per frame with no torch stack and no API key. The small model file is fetched once into the cache and verified against a pinned SHA-256 before it is ever loaded. It returns boxes rather than pixel masks, and any phrase other than face is skipped with a printed note. On the public showcase scenes it matched or slightly exceeded the SAM face counts at a fraction of the compute, which is the right trade for a quick share-safe pass. The summary records the box mask source, so a fast-tier run never presents itself as a full SAM run.

## Privacy notes

API and hosted modes send the clip to the endpoint you chose. Local mode and the local page send nothing. SAM results are cached on your machine under your home cache folder so re-renders skip repeat uploads for the same clip and phrase. Use --no-cache to force a fresh pass. Output files are written next to your input with a redacted suffix and never overwrite the original.

## Honest limits

Audio defaults to keep, so the original voices ship unless you choose otherwise. --audio mute removes the track entirely. --audio pitch shifts voices with pitch factor 0.8 by default while keeping tempo and duration, so speech stays understandable but no longer sounds like the real speaker. If pitch shifting fails the run writes no audio rather than leaking the original track.

--speech-pii scans what is said, not just how it sounds. The audio is transcribed locally with faster-whisper (install with `pip install "open-redactor[speech]"`), or you can pass --transcript with a Whisper JSON file from any other tool. Spoken digit sequences are rebuilt from number words and digit tokens and verified before anything is muted: Luhn for card numbers, SSA issuing rules for Social Security numbers. Matched spans are muted with a small margin and listed in the summary with their severity. Asking for the layer explicitly without a transcriber is an error with the fix line, while --sensitive enables it only when a transcriber is present and says so when it skips.

Metadata is handled on every run. The source is inspected for GPS location, creation time, device info, titles, and embedded text, and the output is written without those tags: renders pin the strip explicitly rather than relying on encoder defaults, and the finished file is checked before the run reports success. The summary records what was found and whether it was stripped. --keep-metadata carries the source tags onto the output deliberately for archival workflows, and the findings then appear as elements in the summary, since they remain in the file you would share. Shadow audits list metadata findings as elements too, so the audit gate can fail on a tagged original before anyone shares it by mistake.

The pixel layer gets its own check with --location-audit. Metadata can be stripped completely, but the scene itself still carries location: signage, architecture, vegetation, terrain, and road furniture survive redaction because they are not objects anyone named. With the flag on, a geolocation model on the API (standard tier only) tries to place the processed photo, one sampled frame of a processed video, or the source in a shadow audit, and the summary gains a location_audit block with the estimated placement, the model's stated chance that the placement lands within 25 km, and the cues it used. The standing note in every block: redaction reduces locatability, it does not remove it. The flag is off by default because it sends the output to the API, it refuses local backends and local providers, and a failed estimate is recorded as failed in the summary rather than skipped silently. The text layer also covers GPS coordinate stamps burned into the frame itself, the dashcam overlay kind, as a "gps coordinates" PII kind under --pii-text and --sensitive.

Preview renders only the first 3 seconds. A clean preview does not prove the whole clip, the coverage report on the full run is the audit.

Pixel-perfect masks need Node and @meta-sam/parser on the machine. Without them the run falls back to box masks with padding and carry, which is safe but less tight. The summary JSON records which one a run actually used as `sam_mask_source`, so the claim is checkable instead of assumed.

Blur strength adapts to region size. The default kernel is sized for faces, and a large plate or sign blurred at that strength can stay readable, so bigger regions are rendered in their own layer with a kernel scaled to their size. When setup is in doubt, run `open-redactor doctor`: it checks ffmpeg, the mask decoder, OCR engines, the optional extras, and whether the API key env var is set, and prints the install line for anything missing. It never prints the key itself.

Detection can miss. Padding, carry-forward, and gap flags reduce the leak risk, and the contact sheet is your final human check. Treat any NEEDS REVIEW line in a coverage report as a stop before sharing.

Video input formats are MP4, MOV, MKV, WebM, AVI, and M4V. Photo inputs are JPG, PNG, WebP, BMP, and HEIC or HEIF, and a photo run writes a redacted photo in the same format using the same targets, presets, and sensitive layers. HEIC is what iPhones shoot by default, and it needs the one-time extra `pip install "open-redactor[heic]"` because OpenCV cannot decode it. Phone photo orientation is applied on read, and a HEIC file is transcoded to a temp PNG for detection only, so the API never sees a format it would reject and your original is never changed. Phones shoot MOV, screen tools emit WebM and MKV, and older cameras write AVI, so all of them are accepted and batch mode picks them all up. Batch mode also picks up photos and audio files in the same folder and runs each through its own pipeline. Output is always MP4 for share compatibility. Non-MP4 inputs sent to the API or a hosted endpoint are transcoded to a temp MP4 for the upload only, your original file is never changed.

## Sensitive documents

Use --preset documents --pii-text for passports, bank cards, driver licenses, and paperwork. The preset blurs the physical objects. The text layer blurs printed numbers and addresses found by OCR, with Luhn verification for card numbers so order numbers and timestamps stay untouched. A passport held to camera is covered as an object even when its number is too small to read, and a card number typed on a screen share is covered as text even with no physical card in frame. You need both layers for document work.

## Codes, location, and badges

--codes covers QR codes and barcodes, --preset location covers street signs, house numbers, plates, and mailboxes, and name badges and lanyards are part of the documents preset. --sensitive turns on the text and codes layers together and pairs well with the documents and location presets.

## Analytics, shadow, and replacement

Normal runs print analytics and write a .summary.json. --shadow runs detection only and writes an .audit.txt with severity per element, so start with a shadow audit on anything sensitive or unfamiliar. Shadow audits work on photos too, with detections, text and code hits, and metadata findings listed as elements and nothing rendered. --mode replace swaps regions for generated stand-ins instead of masking, with per track deterministic fakes. Shadow first, replace when the natural feel matters, blur when it does not.

## Audio only files

MP3, WAV, and M4A files are inputs in their own right, for voice notes, dictated messages, call recordings, and podcast clips that never had a picture. The output keeps the input format and the usual safe naming, and the run covers the three channels an audio file carries.

Spoken PII works exactly as it does on a video track. `--speech-pii` transcribes locally with the speech extra, or `--transcript` takes a Whisper JSON with word timings, and verified card numbers, SSNs, phone numbers, and emails are muted in place and reported as `speech:<kind>:<n>` elements in the summary. `--sensitive` enables the layer when a transcriber is present. The whole track modes apply too: `--audio mute` writes a silent copy, and `--audio pitch` with `--pitch-factor` disguises the voice with tempo preserved.

Container tags get the same treatment as video. ID3 and MP4 tags are inspected before the run, stripped from the output by default with the findings reported in the summary metadata block, and kept only with `--keep-metadata`. A shadow audit with `--shadow` renders nothing and lists speech hits and metadata findings as elements, so the audit gate can fail on a tagged or revealing original before anyone shares it.

## Keep or exclude a track

Every track has a key, printed in the coverage report and the summary JSON, for example `person:0`, `face:1`, or `pii:email:2`. Two flags use those keys. `--keep person:0` leaves that one track visible while everything else is redacted, which is the family video case where one person is fine being seen. `--exclude person:1` drops a named track from redaction, which is the fix for a false positive. Both flags repeat, both accept globs such as `person:*`, and both apply to photos as well as video. A pattern that matches nothing prints a warning with the track keys that do exist. Whatever was left visible is listed in the coverage report and under `kept_visible` in the summary JSON, so an exception never hides in the record.
