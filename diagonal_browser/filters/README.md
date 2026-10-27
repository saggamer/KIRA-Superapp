# Bundled ad and tracker filters

`ads-and-tracking.snapshot.json.gz` is an offline snapshot built with the
Ghostery Adblocker engine. It includes EasyList, EasyPrivacy, Peter Lowe's list,
and the uBlock-compatible badware, privacy, ad, resource-abuse, quick-fix, and
unbreak subscriptions exposed by `@ghostery/adblocker`'s ads/tracking preset.
It does not include Acceptable Ads or cookie-consent removal lists.

The gzip contains JSON with the full original source list texts, their exact
download URLs and original notices, the resource/scriptlet JSON, the build
timestamp, rule counts, engine format version, and a compiled engine. The source
texts are retained so the bundled rules can be inspected and rebuilt; each list
retains its own upstream license. Ghostery's engine and adapters are MPL-2.0;
their license is included in the installed npm packages. Upstream source and
licensing information: <https://github.com/ghostery/adblocker> and
<https://easylist.to/>.

Rebuild the distributed snapshot with `node update_adblock_assets.mjs` from the
browser directory. Runtime updates use the same curated URLs and write only to
the user's profile, keeping this bundled fallback available. An incomplete or
failed download never replaces the active snapshot.
