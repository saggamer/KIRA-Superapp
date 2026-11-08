# Finder follow-ups and image analysis

Image analysis no longer has the 120-second subprocess cutoff or the surrounding
135-second bridge cutoff. Its bridge runs off the UI thread and waits for the
actual result. Time spent waiting for image analysis is excluded from the v1.5
task deadline so a subsequent Finder action can still run. Other tool timeouts
remain unchanged.

A bare `open it` no longer produces an artifact error when no generated artifact
exists. An older artifact is opened through that shortcut only when the latest
assistant response actually refers to its path. Unresolved app/project references
remain with the model and recent conversation.

The desktop harness describes local file/app capabilities, how to locate a
cloned project's existing folder, and how to open that folder in Finder. Generic
claims that all local access is unavailable trigger a corrective model turn for
an actual user action. Questions, mentions, negations and real permission/tool
failures do not trigger that correction. A cloned repository is not treated as
an installed app or executed merely to locate its folder.

Validation in the clean GitHub checkout: 131 tests ran successfully, with one existing skip.
The wait checks simulate a long image-analysis duration and verify that the
remaining task can still complete; they do not wait 10,000 real seconds or test
image-recognition quality. Finder execution is simulated in regression checks.
