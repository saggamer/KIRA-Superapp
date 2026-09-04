# Isolated KIRA UI verification

## Environment

Actual ui.html running in the Codex browser, connected over loopback HTTP to
the real KiraBrain API. scripts/isolated_ui_check.py provides this test-only
transport. Data and HOME are disposable; Keychain service names are isolated.
This is not the native WKWebView desktop runtime or a virtual macOS machine.
No paid API calls, production chat edits, or external app launches were made.

## Live passes

- Startup creates a real chat and displays it in history.
- Attachment dropdown opens; Choose files opens a file chooser from its click.
- DOCX upload, then simultaneous PDF and DOCX upload succeed through the UI.
- All three stored files are byte-identical to their original fixtures.
- Removing an attachment updates the composer tray.
- Composer is focused at startup and after upload; caret is visible with
  computed color rgb(244, 244, 245), and text selection is enabled.
- Escape closes the attachment dropdown and restores composer focus.
- Chat and Code workspace switches work; Code lists installed IDEs and providers.
- Profile menu opens Settings; Providers, Model Library, MCP Servers, and Plugins
  tabs render actual isolated backend state.
- Sources opens and closes, reporting no sources for a new chat.
- KIRA Menu opens with backend inventory and the uploaded files.
- Browser error/warning log snapshot at the end is empty.

## Regression passes

- 131 Python unittest tests pass. These include real artifact serialization as
  well as mocked model/network tests; they are not 131 live feature checks.
- JavaScript clipboard suite passes, including new synchronous-picker and
  focus-preservation checks.

## Not verified live

- Native macOS Finder paste and WKWebView file dialog. Browser automation's
  text fill triggered the native-paste fallback; this was explicitly denied by
  the test transport and shown as an error, not reported as a successful paste.
- Temporal caret blinking in WKWebView; browser focus and visible caret were checked.
- Orchestrator inference, complete research-to-artifact tasks, coding model calls,
  paid-provider billing, Keychain reveal, plugin execution and MCP tool sessions.
- Microphone, speech models, OS automation, and launching files in external IDEs.

The isolated runtime has no configured model or credentials. Its startup reports
Orchestrator is not loadable; therefore this run cannot certify all features.
Its later generic "Orchestrator V1 active" console banner is not proof of loading.
Restart the desktop app to load the UI and text-selection fixes.
