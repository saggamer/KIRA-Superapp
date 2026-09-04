# KIRA Plugins

KIRA discovers external plugins from JSON manifests in this folder. Installing a
manifest copies its declared entry point into an isolated plugin directory and
leaves the plugin disabled until it is explicitly enabled.

Example manifest:

```json
{
  "id": "example-tools",
  "name": "Example Tools",
  "version": "0.1.0",
  "description": "Adds a focused set of local tools.",
  "entrypoint": "plugin.py"
}
```

The entry point must be a `.py` file next to the source manifest. KIRA never
stores secrets in plugin manifests. Enabling or disabling an external plugin
takes effect after KIRA restarts.
