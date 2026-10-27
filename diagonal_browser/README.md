# Diagonal Browser for Kira

Diagonal Browser is a local, persistent Playwright MCP server for Kira models. It uses the Antigravity bridge's `ddgs` search tool for live search and the installed Google Chrome for page visits, interactions, screenshots, and WebM screen recordings. Browser artifacts are saved under `diagonal_browser/artifacts/` and kept out of Git.

## Setup

```bash
cd diagonal_browser
npm ci
npm run setup
```

Node 20 or newer and Google Chrome are required. The setup command installs Playwright's small FFmpeg runtime for video; the system Chrome binary is reused. If the bridge or Chrome is elsewhere, set `DIAGONAL_DDGS_BIN` and `DIAGONAL_CHROME_BIN`. `DIAGONAL_HEADED=1` shows the browser window. `DIAGONAL_ARTIFACTS_DIR` changes the output folder.

On macOS, double-click `SETUP_DIAGONAL_BROWSER.command` in the project root for the same setup with prerequisite checks.

## Desktop browser

Click **DIAGONAL** in Kira Superapp's Workspace sidebar, double-click `OPEN_DIAGONAL_BROWSER.command` in the project root, or run `npm run desktop` here. Repeated launches focus the existing browser window. This opens a full Diagonal Browser window with tabs, an address bar, navigation controls, live DDG search results, page screenshots, and WebM recording. The new tab page uses the generated black hole artwork in `ui/assets/black-hole.png`; the toolbar wordmark and macOS Dock icon use black DIAGONAL letters on white. Type a web address to open a site, or type search words to use the bridge's `ddgs` engine. Screenshots and recordings are saved in `artifacts/`; the capture notification has a **Show file** button.

Desktop recording uses `ffmpeg` from your `PATH` (or `DIAGONAL_FFMPEG_BIN`) and records the active page at 5 frames per second without audio. Keep the recorded tab selected until you stop. The desktop window and Kira's MCP connection are separate browser sessions.

The desktop uses a compact two-row toolbar with consistent SVG icons: navigation and the address bar on the left, image search and page tools on the right. Secondary tools stay visible as icon buttons with accessible labels and tooltips; controls adapt down to the minimum 800px window width. The native page starts immediately below the 104px toolbar. The home page keeps the black hole artwork, adds local recent searches from Kira and the browser, and restores those results without another search. Reloading a results page runs a fresh search. The selected tool is highlighted, and Escape closes a tool panel.

Keyboard shortcuts include **Cmd/Ctrl+L** for the address bar, **Cmd/Ctrl+T** for a new tab, **Cmd/Ctrl+Shift+T** to reopen a closed tab, **Cmd/Ctrl+Shift+H** for history, and **Cmd/Ctrl+1–9** to select a tab (9 selects the last). Up to 20 closed pages/searches can be reopened during the desktop session. Reduced-motion preferences disable interface animations.

Run `npm test` for browser/MCP unit and integration checks. On macOS, `npm run test:desktop` launches an isolated temporary profile and local search/page fixtures to verify the 1440px/800px layouts, native page alignment, history restoration, fresh search reload, closed-tab reopening and tool panels. Preview screenshots are written to the ignored `artifacts/` directory; the test profile is removed afterward.

## Search history for Kira and other models

Open **History** in the toolbar (or **Cmd/Ctrl+Shift+H**) to search saved queries, result titles, URLs, snippets, and cached page text. Filter by searches/pages or Kira/browser. **Read saved** reads the local copy; **Restore results** restores a search without rerunning it; **Open website** visits the current source. Saved results show when they were captured. Reading a saved search does not mean its linked pages were read.

The desktop, MCP browser, and Kira's regular `WEB_SEARCH`, `WEB_RESEARCH`, image-reference searches, and `WEB_FETCH` tools write to the same local `.profile/search_history/` directory. `DIAGONAL_PROFILE_DIR` changes this directory for all three paths. Records survive restarts and are stored as separate atomic JSON files with owner-only permissions. History starts with this feature; older searches are not reconstructed. Completed search results and captured text are retained locally; there is no automatic expiry. Page captures are limited to 25,000 characters, with truncation identified in the saved record; ordinary web research may save a shorter excerpt. Form field values are not explicitly collected, the visible Chrome login handoff is excluded, and known authentication query parameters are stripped from stored URL fields. Cached page content can still contain personal information visible on the page, so `.profile/` stays out of Git.

Models have three MCP tools:

| Tool | Purpose |
| --- | --- |
| `search_history` | Search local history with `query`, `kind` (`all`, `search`, `page`), `source` (`all`, `llm`, `desktop`), `limit`, and `offset`. Returns `historyId`, timestamps, snippets, and `nextOffset`. |
| `read_history` | Read a `historyId` without opening a browser or using the network. Returns up to 5,000 text characters or 10 search results; continue with `textOffset`/`resultOffset`. |
| `revisit_history` | Open a saved page live, or use `resultIndex` to open a result from a saved search. With no result index, a search returns its cached result list. |

For example, Kira can retrieve earlier research with:

```text
[BROWSER_TOOL]
ACTION: search_history
QUERY: orchid growing guides
SOURCE: llm
KIND: search
[/BROWSER_TOOL]
```

Then use `ACTION: read_history` with `HISTORY_ID` from the result, or `ACTION: revisit_history` with `HISTORY_ID` and `RESULT_INDEX: 0`. The model is instructed to treat cached website text as untrusted historical evidence and make a fresh visit when current information matters. History retrieval is paginated to fit small model context windows; the native Kira controller and Spark tool schema accept these actions too.

Select **Images** on the home search or in the toolbar to search pictures across the web. Results appear in a visual grid with Large, Wallpaper, Transparent, GIF, and Black & white filters. Select a picture for a larger preview, then copy its image URL, open the original image, or visit its source page in a new tab. Image search runs through an isolated Python environment installed by `SETUP_DIAGONAL_BROWSER.command` using the `ddgs` metasearch package. Set `DIAGONAL_IMAGE_PYTHON` to another Python executable with `ddgs` installed if needed. Live image results depend on the search provider and source sites; some originals may only load through their source page.

The **Library** shows saved PNG screenshots and WebM recordings; select a recording to play it inside the browser. **Shield** opens the ad blocker's settings, per-site controls, rule editor, and blocked-request log. **Inspect** shows URL structure and page metadata, headings, links, images, forms, and readable text. It can copy or save structured JSON for an LLM. Kira can request the same structured data from its Playwright session with the `inspect_page` MCP tool. The desktop inspector and blocker use Electron's native APIs; the bridge's Manifest V2 package is a reference template and is not loaded into the desktop window.

## Ad blocking for the browser and models

Diagonal uses the maintained [Ghostery Adblocker engine](https://github.com/ghostery/adblocker) with a bundled offline ads/tracking snapshot: EasyList, EasyPrivacy, Peter Lowe's list, and uBlock-compatible ad, badware, privacy, quick-fix and unbreak filters. The initial snapshot includes more than 120,000 network rules and 36,000 cosmetic rules. Standard network filtering, redirects, exceptions, CSS element hiding, and supported scriptlets are active by default. Direct main-page navigation is allowed. Supported lists are broad protection across sites; they cannot guarantee removal of every advertisement, particularly server-inserted video ads. Procedural HTML filtering and extended selector rules are not implemented in these adapters. Cookie-consent removal lists are not enabled.

Protection covers desktop pages and popups, and Kira's ordinary tabs, popups, visible login handoff and recording contexts. Playwright contexts block service workers so worker-owned requests cannot silently bypass request interception. Its cosmetic monitor watches new DOM features, including late ad slots, and uses constructed stylesheets without weakening website CSP. Electron uses Ghostery's native session adapter. No extension installation is required.

Click **Shield** to enable/pause globally, pause a website, enable/disable cosmetic cleanup, add/remove a custom rule, explain a resource match, or update lists. Site exceptions cover subdomains; a more specific exception takes precedence. Reload affected pages after changes, because injected styles and scriptlets already on an open page cannot be fully undone in place. The previous desktop global toggle is migrated on first use.

Settings and custom filters are stored under `.profile/adblock/` with owner-only permissions and are shared between desktop and model sessions. Atomic writes and a cross-process lock preserve simultaneous edits. Blocked-request logs contain at most 100 recent URLs, resource types, and matched rules in memory, with sensitive authentication query keys removed; request bodies and credentials are not recorded. Counters/logs belong to the current browser process and reset on restart. Model observations include a compact protection indicator and blocked-request count.

Lists older than 24 hours refresh in the background on startup and on an hourly check. Failed updates retain the last working snapshot and are retried no more than hourly. **Update lists** and `adblock_update` force a refresh from the same curated URLs; models cannot supply an arbitrary subscription endpoint. `DIAGONAL_ADBLOCK_AUTO_UPDATE=0` disables background downloads for offline/test runs. Raw source list texts, URLs, original license notices, resources, and the compiled engine are retained in the bundled snapshot; see [filter provenance](filters/README.md).

| MCP tool | Model control |
| --- | --- |
| `adblock_status` | Read settings, rule counts, update time, site state and recent blocked requests; optional `url` scopes site details. |
| `adblock_set` | Set `enabled`, `cosmetic`, `autoUpdate`, or `site` plus `siteEnabled`. |
| `adblock_rule` | Add/remove one `rule` with `operation: add/remove`, such as `||ads.example.com^` or `example.com##.advertisement`. |
| `adblock_check` | Explain `url` on `pageUrl` for `resourceType`, without visiting it or changing counters. |
| `adblock_update` | Refresh all curated lists while keeping the last good snapshot on failure. |

Kira uses the same `[BROWSER_TOOL]` bridge. For example:

```text
[BROWSER_TOOL]
ACTION: adblock_check
URL: https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js
PAGE_URL: https://example.com/
RESOURCE_TYPE: script
[/BROWSER_TOOL]
```

To pause one site, use `ACTION: adblock_set`, `SITE: example.com`, and `SITE_ENABLED: false`; then reload. Re-enable it with `SITE_ENABLED: true`. Kira's native tool schema, browser controller and orchestral candidate accept all five controls. The model is instructed to prefer site exceptions when troubleshooting and to ignore website instructions to disable protection.

An optional unpacked Chrome extension is in `extensions/url-inspector/`. In Chrome, open `chrome://extensions`, turn on Developer mode, choose **Load unpacked**, and select that folder. Its popup inspects only the active page when opened, then copies compact JSON for an LLM. The desktop browser already has its own **Inspect** button, so this extension is useful mainly in a separate Chrome window.

Restart Kira after setup. Kira's `[BROWSER_TOOL]` action now sends calls to this MCP server and keeps one server process alive for the Kira session. Existing `WEB_SEARCH` and `WEB_BROWSE` actions still work. A local draft config for other MCP hosts is in `kira_mcp_connectors/diagonal_browser.json`. Run `npm run mcp-config` to print a config with the current absolute path for another machine.

For model driven browser control, `navigate`, `observe`, and actions return the current URL, title, readable text, and up to 80 visible links and controls with indices. `act` performs one interaction and returns the updated view in the same call. It accepts an exact accessible `TARGET_ROLE` and `TARGET_NAME`, an `INDEX` from the latest observation, a CSS `SELECTOR`, or exact visible `TEXT` for clicks. The returned `observationId` can be supplied as `OBSERVATION_ID` with an index or coordinates; a stale ID is rejected. Models should prefer role and name for controls, then inspect the refreshed view before the next step. Form actions include fill, check, uncheck, and select_option. Browser results are projected into bounded, valid JSON so controls remain visible to the model on text-heavy pages. Password, token, and email input values are masked in observations. Use `SCREENSHOT: true` with `observe` to save a viewport PNG and get its path. The Kira MCP session controls its own persistent Chrome context; the standalone Diagonal desktop window remains a separate session.

`observe` also reports each control's viewport `x` and `y`, bounding box, and cell in an 8-row by 12-column grid. `SCREENSHOT: true` draws the grid, row/column labels, and numbered element outlines onto the saved PNG. `act` can click an `X` and `Y` pair from that observation when a semantic locator is unavailable. Kira's local text model receives the grid and element metadata; the PNG is an artifact for visual review or a vision-capable host, not automatically decoded by the text model.

The browser's **Accounts** panel saves up to three nickname/email pairs in `.profile/email_profiles.json` with owner-only file permissions. Click **Use on page** on a sign-in page to fill its visible email/username field. Kira can call `list_email_profiles` and `use_email_profile` with `NICKNAME`. This does not store passwords, submit a form, or bypass the site's password, passkey, OAuth, or verification flow. An email address alone cannot authenticate to arbitrary websites.

## OAuth, passkeys, and saved sessions

Diagonal's desktop browser preserves real popup windows and their `window.opener` relationship, so websites that support in-browser OAuth popups can return to the page that launched them. A website can also navigate through a redirect flow in the main tab. The desktop browser uses its persistent Electron session for site cookies. Some identity providers reject embedded browsers: [Google's OAuth policy](https://developers.google.com/identity/protocols/oauth2/policies) disallows its authorization endpoint in an embedded user agent, so a working local popup does not guarantee every provider will sign in inside Electron.

Kira's MCP browser tracks OAuth popups as tabs. Use `begin_login` with a site URL to open a visible Chrome window for a user to complete OAuth, a password, or an OS passkey prompt. Kira can `observe` and `act` on that window. After the user completes the site's flow, `finish_login` imports cookies, local storage, and IndexedDB into Kira's browser context and saves them in `.profile/kira_auth_state.json` with owner-only permissions. This file contains sensitive login tokens; keep `.profile/` private and out of Git. `finish_login` with `SAVE: false` closes the login window without importing it. Kira never receives the passkey private key or password through these tools. The desktop and Kira sessions are separate; a login completed in one does not silently sign in the other.

On macOS, a packaged and properly signed Diagonal build can enable native Touch ID WebAuthn by setting `DIAGONAL_WEBAUTHN_KEYCHAIN_GROUP` to a group present in the app's `keychain-access-groups` entitlement before launching. The browser then offers a native chooser when multiple discoverable passkeys match. The unsigned development Electron app cannot provide Touch ID platform passkeys through this hook. A visible Chrome login through `begin_login` is the supported route for the user's existing OS or browser passkeys during development. Electron's [WebAuthn configuration requirements](https://www.electronjs.org/docs/latest/api/app#appconfigurewebauthnoptions-macos) describe the signed build requirement.

## Kira actions

```text
[BROWSER_TOOL]
ACTION: search
QUERY: Kira Superapp
LIMIT: 5
[/BROWSER_TOOL]

[BROWSER_TOOL]
ACTION: navigate
URL: https://example.com
[/BROWSER_TOOL]

[BROWSER_TOOL]
ACTION: observe
SCREENSHOT: true
[/BROWSER_TOOL]

[BROWSER_TOOL]
ACTION: scroll
PIXELS: 700
[/BROWSER_TOOL]

[BROWSER_TOOL]
ACTION: use_email_profile
NICKNAME: Personal
[/BROWSER_TOOL]

[BROWSER_TOOL]
ACTION: begin_login
URL: https://example.com/login
[/BROWSER_TOOL]

[BROWSER_TOOL]
ACTION: finish_login
SAVE: true
[/BROWSER_TOOL]

[BROWSER_TOOL]
ACTION: act
OPERATION: click
TARGET_ROLE: link
TARGET_NAME: Learn more
[/BROWSER_TOOL]

[BROWSER_TOOL]
ACTION: act
OPERATION: fill
INDEX: 0
OBSERVATION_ID: observation-token-from-previous-view
VALUE: Diagonal Browser
[/BROWSER_TOOL]

[BROWSER_TOOL]
ACTION: screenshot
NAME: example.png
[/BROWSER_TOOL]

[BROWSER_TOOL]
ACTION: start_recording
[/BROWSER_TOOL]

[BROWSER_TOOL]
ACTION: stop_recording
NAME: visit.webm
[/BROWSER_TOOL]
```

Actions: `search`, `observe`, `act`, `list_email_profiles`, `use_email_profile`, `begin_login`, `finish_login`, `navigate`, `new_tab`, `list_tabs`, `select_tab`, `back`, `forward`, `reload`, `snapshot`, `inspect_page`, `scroll`, `wait_for`, `click`, `fill`, `press`, `screenshot`, `start_recording`, `stop_recording`, `close_tab`. `navigate` and `new_tab` return a `tabId`. Supply `TABID` for a specific tab, or omit it to use the active tab. `act` accepts `OPERATION: navigate|new_tab|select_tab|close_tab|back|forward|reload|scroll|click|fill|press`. `click` accepts a CSS `SELECTOR`, exact visible `TEXT`, or an `INDEX` from the latest snapshot. `fill` accepts `SELECTOR` or `INDEX` plus `VALUE`. `press` accepts `KEY` such as `Enter`. `scroll` accepts `PIXELS` from -1600 to 1600 (positive down, negative up, zero excluded), defaults to 600, and returns `visibleText`, `scroll.y`, `scroll.maxY`, `scroll.atEnd`, and `moved`. Use those fields to read lower content and stop at the page boundary. Screenshots accept `NAME` ending in `.png`; recordings accept `NAME` ending in `.webm`.

Recording starts a new tab in a video enabled browser context and returns its `tabId`. Continue browser actions in that tab until `stop_recording`. The source tab remains open; after stopping, it becomes active again at its original `sourceUrl`. The result also includes `lastUrl` from the recording. Recording actions do not update the source tab's page state. The video is finalized when the recording context closes.

## Standalone MCP

Run `npm start` in this folder and connect over stdio. Only MCP JSON-RPC is written to stdout. Each host starts its own isolated browser session. The tools return JSON text; `screenshot` and `stop_recording` return absolute artifact paths.

The Antigravity bridge is a file based asynchronous collaboration channel. A request for tool review was posted to its `CHATGPT_INBOX.md`; Antigravity's response, if any, will appear in `ANTIGRAVITY_INBOX.md`. The MCP server does not require Antigravity to be running.
