"""Auditable channel gate for compatible v1.5 adapter checkpoints.

The gate only chooses which already trained adapter handles a request.  Tool
execution and permission checks remain in Kira's action handlers.
"""
import json
from pathlib import Path
import re


APP_CONTROL = re.compile(
    r"\b(?:installed apps?|available apps?|app integration|integration context|"
    r"subagents?|specialist agents?|registered agents?|agent registry|"
    r"queue (?:a |the |this )?task|run (?:the |this )?agent|"
    r"mcp|connectors?|configuration targets?)\b", re.I)
BROWSER = re.compile(
    r"\b(?:diagonal browser|browser|web ?page|tab|tabs|screenshot|"
    r"web search|search (?:the )?web|search online|look up online|ad ?block(?:er|ing)?|privacy shield|block ads|"
    r"search history|browsing history|previous searches|past searches|earlier searches|"
    r"previously searched|searched (?:for )?(?:before|earlier|yesterday)|saved searches|"
    r"surf (?:the )?(?:web|youtube)|browse (?:the )?(?:web|youtube)|"
    r"search (?:on |through )?youtube|youtube (?:video|search)|"
    r"fresh observation|button|textbox|combobox|click|fill|select option)\b", re.I)
DESKTOP_BROWSER_OPEN = re.compile(
    r"\b(?:open|launch|start)\b.{0,35}\b(?:a|an|any|my|the)?\s*(?:web\s+)?browser\b|"
    r"\b(?:open|launch|start)\s+(?:safari|chrome|firefox|edge)\b", re.I)
MCP_CONTROL = re.compile(r"\b(?:mcp|connectors?|configuration targets?)\b", re.I)
BROWSER_HISTORY = re.compile(
    r"\b(?:search|browsing|browser) history\b|"
    r"\b(?:previous|past|earlier|saved) (?:searches|research|pages)\b|"
    r"\bpreviously searched\b|\bsearched\b.{0,40}\b(?:before|earlier|yesterday|recently)\b|"
    r"\bwhat (?:did|have|has|had) (?:you|we|kira) (?:search(?:ed)?|read|visit(?:ed)?)\b", re.I)


def manifest_root(manifest_path, manifest, runtime_root):
    """A bundled manifest resolves assets beside itself; legacy paths keep their root."""
    return (Path(manifest_path).resolve().parent if manifest.get('bundle_format') == 'kira-v15-v1'
            else Path(runtime_root))


def channel_for_request(request):
    """Choose the next action channel from the user's request, never page text."""
    # A deck about a browser, app, or webpage is still an artifact-control task.
    if re.search(r'\b(?:create|make|build|draft|generate|prepare|need|want|produce)\b.{0,100}'
                 r'\b(?:pptx?|powerpoint|presentation|slide deck|slides?)\b', request, re.I):
        return 'control'
    if re.search(r'(?<!\w)\$browser-(?:research|interaction|history)\b', request):
        return 'browser'
    if (DESKTOP_BROWSER_OPEN.search(request) and not re.search(
            r'https?://|\b(?:diagonal browser|search|navigate|web page|screenshot|click|fill|select option)\b',
            request, re.I)):
        return 'control'
    if re.match(r'^\s*(?:use|open|navigate|browse)\b.{0,35}\b(?:diagonal browser|browser)\b',
                request, re.I):
        return 'browser'
    if BROWSER_HISTORY.search(request):
        return 'browser'
    if APP_CONTROL.search(request):
        return 'control'
    if BROWSER.search(request) or re.search(r'https?://', request):
        return 'browser'
    return 'control'


class AdapterRouter:
    def __init__(self, control_checkpoint, browser_checkpoint, presentation_checkpoint=None,
                 mcp_checkpoint=None, coding_checkpoint=None):
        # Preserve the checkpoint's configured directory: model bytes may be
        # symlinked to the external SSD while architecture.json stays beside
        # the symlink in the project.
        self.paths = {'control': Path(control_checkpoint).absolute(),
                      'browser': Path(browser_checkpoint).absolute()}
        if presentation_checkpoint:
            self.paths['presentation'] = Path(presentation_checkpoint).absolute()
        if mcp_checkpoint:
            self.paths['mcp'] = Path(mcp_checkpoint).absolute()
        if coding_checkpoint:
            self.paths['coding'] = Path(coding_checkpoint).absolute()
        architectures = []
        for path in self.paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
            architectures.append(json.loads((path.parent / 'architecture.json').read_text()))
        compatible = ('base_snapshot', 'layers', 'routed_experts_per_layer',
                      'selected_per_token', 'preserve_top2_residual',
                      'legacy_experts_for_primary_route')
        if any(architectures[0].get(key) != architecture.get(key)
               for architecture in architectures[1:] for key in compatible):
            raise ValueError('adapter architectures are incompatible')
        self.architecture = architectures[0]
        self.active = None

    def activate(self, model, request):
        from .presentation_controller import is_presentation_request
        channel = ('presentation' if 'presentation' in self.paths and
                   is_presentation_request(request) else channel_for_request(request))
        if channel == 'control' and 'coding' in self.paths:
            from .coding_agent_controller import is_coding_agent_request
            if is_coding_agent_request(request):
                channel = 'coding'
        if channel == 'control' and 'mcp' in self.paths and MCP_CONTROL.search(request):
            channel = 'mcp'
        if channel != self.active:
            import mlx.core as mx
            model.load_weights(str(self.paths[channel]), strict=False)
            mx.eval(model.parameters())
            model.eval()
            self.active = channel
        return channel
