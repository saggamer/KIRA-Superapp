#!/usr/bin/env python3
"""Click-through macOS status overlay for KIRA Live and agentic activity."""

import json
import os
import sys
import time
import objc


STATE_PATH = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else ""


try:
    from AppKit import (
        NSApplication,
        NSAttributedString,
        NSBackingStoreBuffered,
        NSBezierPath,
        NSColor,
        NSEvent,
        NSFont,
        NSFontAttributeName,
        NSForegroundColorAttributeName,
        NSScreen,
        NSStatusWindowLevel,
        NSView,
        NSWindow,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowStyleMaskBorderless,
    )
    from Foundation import NSObject, NSMakePoint, NSMakeRect, NSTimer
except Exception:
    raise SystemExit(0)


LABELS = {
    "live": "KIRA Live",
    "listening": "KIRA is listening",
    "thinking": "Orchestrator is thinking",
    "speaking": "KIRA is speaking",
    "agentic": "KIRA is working",
}


def read_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as state_file:
            value = json.load(state_file)
        return str(value.get("state", "off")).strip().lower()
    except Exception:
        return "off"


class OverlayView(NSView):
    state = "off"

    def isOpaque(self):
        return False

    def drawRect_(self, _dirty_rect):
        if self.state == "off":
            return
        bounds = self.bounds()
        accent = NSColor.colorWithCalibratedWhite_alpha_(0.86, 0.82)
        dim = NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.88)

        strip_height = 3.0 if self.state in {"live", "listening"} else 5.0
        accent.setFill()
        NSBezierPath.fillRect_(NSMakeRect(0, 0, bounds.size.width, strip_height))

        pointer = NSEvent.mouseLocation()
        screen = NSScreen.mainScreen().frame()
        x = max(12.0, min(bounds.size.width - 226.0, pointer.x - screen.origin.x + 18.0))
        y = max(14.0, min(bounds.size.height - 52.0, pointer.y - screen.origin.y - 46.0))
        badge = NSMakeRect(x, y, 208.0, 34.0)
        dim.setFill()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(badge, 17.0, 17.0).fill()

        dot = NSMakeRect(x + 12.0, y + 13.0, 8.0, 8.0)
        accent.setFill()
        NSBezierPath.bezierPathWithOvalInRect_(dot).fill()

        text = LABELS.get(self.state, "KIRA Live")
        attributes = {
            NSForegroundColorAttributeName: NSColor.colorWithCalibratedWhite_alpha_(0.93, 0.94),
            NSFontAttributeName: NSFont.systemFontOfSize_weight_(12.0, 0.45),
        }
        NSAttributedString.alloc().initWithString_attributes_(text, attributes).drawAtPoint_(
            NSMakePoint(x + 29.0, y + 9.0)
        )


class OverlayController(NSObject):
    def initWithView_window_(self, view, window):
        self = objc.super(OverlayController, self).init()
        if self is not None:
            self.view = view
            self.window = window
            self.last_state = None
            self.off_since = None
        return self

    def tick_(self, _timer):
        state = read_state()
        if state != self.last_state:
            self.last_state = state
            self.view.state = state
            if state == "off":
                self.window.orderOut_(None)
                self.off_since = time.monotonic()
            else:
                self.off_since = None
                self.window.orderFrontRegardless()
        if state != "off":
            self.view.setNeedsDisplay_(True)
        elif self.off_since and time.monotonic() - self.off_since > 180.0:
            NSApplication.sharedApplication().terminate_(None)


def main():
    app = NSApplication.sharedApplication()
    screen = NSScreen.mainScreen().frame()
    window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        screen,
        NSWindowStyleMaskBorderless,
        NSBackingStoreBuffered,
        False,
    )
    window.setOpaque_(False)
    window.setBackgroundColor_(NSColor.clearColor())
    window.setHasShadow_(False)
    window.setIgnoresMouseEvents_(True)
    window.setLevel_(NSStatusWindowLevel)
    window.setCollectionBehavior_(
        NSWindowCollectionBehaviorCanJoinAllSpaces
        | NSWindowCollectionBehaviorFullScreenAuxiliary
    )
    view = OverlayView.alloc().initWithFrame_(NSMakeRect(0, 0, screen.size.width, screen.size.height))
    window.setContentView_(view)
    controller = OverlayController.alloc().initWithView_window_(view, window)
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        1.0 / 30.0,
        controller,
        "tick:",
        None,
        True,
    )
    app.run()


if __name__ == "__main__":
    main()
