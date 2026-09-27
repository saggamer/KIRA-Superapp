from pathlib import Path
import tempfile
import unittest

from kira_live import (
    CompositeBridgeState,
    EmotionState,
    EmotionTracker,
    KiraLiveCompositePlan,
    KiraLiveCompositeRuntime,
    KiraLiveConfig,
    KiraLiveDuplexController,
    KiraLiveSession,
    ListenerState,
    LiveExpertRouter,
    LiveState,
    PersistentWorkflowMemory,
    Qwen4MicroBlueprint,
    ThinkerSeed,
    ThinkerState,
    VadSegmenter,
    resolve_composite_weights,
)
from kira_live.readiness import kira_live_readiness
from kira_live.mlx_voice import KiraIntegratedQwenVoice, emotion_style_instruction
from kira_live.audio_devices import pick_audio_device
from kira_live.privacy import sanitize_public_output
from kira_live.turn_guard import LiveTurnGuard
from kira_live.token_bus import remap_token_ids
from kira_live.memory_grounding import grounded_recall
from kira_live.checkpoint_bundle import resolve_live_checkpoint


class KiraLiveTests(unittest.TestCase):
    def test_default_configuration_is_valid(self):
        config = KiraLiveConfig()
        config.validate()
        self.assertEqual(config.thinker.model_id, "Qwen/Qwen3.5-0.8B")
        self.assertEqual(config.audio.samples_per_frame, 320)
        self.assertEqual(config.memory.prompt_budget_chars, 48_000)

    def test_native_fallback_vad_accepts_live_320_sample_frames(self):
        from listen import create_vad

        vad = create_vad(prefer_ten=False)
        speech, probability = vad.process([0.0] * 320)
        self.assertFalse(speech)
        self.assertLess(probability, 0.62)
        speech, probability = vad.process([0.004] * 320)
        self.assertFalse(speech)
        self.assertLess(probability, 0.62)

    def test_native_audio_uses_concrete_input_and_output_devices(self):
        devices = (
            {"name": "Microphone", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
        )
        self.assertEqual(pick_audio_device(devices, -1, "max_input_channels"), 0)
        self.assertEqual(pick_audio_device(devices, -1, "max_output_channels"), 1)
        self.assertEqual(pick_audio_device(devices, 1, "max_output_channels"), 1)

    def test_live_turn_guard_rejects_noise_and_rapid_duplicates(self):
        guard = LiveTurnGuard(duplicate_window_seconds=3.0)
        self.assertEqual(guard.accept("啊。", now=1.0), (False, "low_information"))
        self.assertEqual(guard.accept("Can you hear me?", now=2.0), (True, "accepted"))
        self.assertEqual(
            guard.accept("can you hear me", now=3.0),
            (False, "rapid_duplicate"),
        )
        self.assertEqual(guard.accept("can you hear me", now=6.5), (True, "accepted"))

    def test_listener_thinker_token_bus_remaps_pieces_without_text_prompt(self):
        class Source:
            def convert_ids_to_tokens(self, ids):
                return {3838: "What", 374: "Ġis"}.values()

        class Target:
            unk_token_id = None

            def convert_tokens_to_ids(self, pieces):
                table = {"What": 3710, "Ġis": 369}
                return [table[piece] for piece in pieces]

        self.assertEqual(remap_token_ids((3838, 374), Source(), Target()), (3710, 369))

    def test_uncertain_emotion_is_not_presented_as_fact(self):
        state = EmotionState(
            {"neutral": 0.4, "joy": 0.1, "sadness": 0.3, "anger": 0.1, "fear": 0.1, "surprise": 0.0},
            valence=-0.2,
            arousal=0.3,
            confidence=0.2,
            observed_at=0.0,
        )
        self.assertIn("do not infer", state.prompt_context())

    def test_emotion_decays_toward_neutral(self):
        tracker = EmotionTracker(smoothing=0.0, neutral_half_life=1.0)
        tracker.update(
            {"anger": 1.0}, valence=-0.8, arousal=0.9, confidence=0.9, observed_at=10.0
        )
        self.assertGreater(tracker.state(now=20.0).probabilities["neutral"], 0.99)

    def test_router_uses_agentic_and_emotion_experts(self):
        state = EmotionState(
            {"neutral": 0.1, "joy": 0.0, "sadness": 0.7, "anger": 0.1, "fear": 0.1, "surprise": 0.0},
            valence=-0.7,
            arousal=0.6,
            confidence=0.9,
            observed_at=0.0,
        )
        route = LiveExpertRouter().route("Please fix and test this code", state)
        self.assertTrue({"agent_planning", "tool_use"} & set(route.active_experts))
        self.assertIn("emotion_prosody", route.active_experts)
        self.assertAlmostEqual(sum(route.weights.values()), 1.0)

    def test_live_session_supports_barge_in(self):
        session = KiraLiveSession()
        session.start()
        session.transcript_ready("Hello Kira")
        session.response_started()
        event = session.voice_started(0.9)
        self.assertEqual(event.kind, "utterance_started")
        self.assertEqual(session.state, LiveState.LISTENING)
        self.assertTrue(any(item.kind == "barge_in" for item in session.events))

    def test_duplex_listener_keeps_ngrams_and_cancels_brain_and_talker_on_barge_in(self):
        session = KiraLiveSession()
        session.start()
        session.response_started()
        stopped = []
        controller = KiraLiveDuplexController(
            session,
            stop_playback=lambda: stopped.append(True),
        )
        ticket = controller.begin_generation()
        frame = [0.02, -0.02] * (session.config.audio.samples_per_frame // 2)
        for _ in range(5):
            controller.ingest_frame(frame, 0.1)
        events = ()
        for _ in range(controller.segmenter.start_speech_frames):
            events = controller.ingest_frame(frame, 0.95)
        self.assertEqual(events[0].kind, "speech_start")
        self.assertTrue(ticket.cancelled.is_set())
        self.assertEqual(stopped, [True])
        self.assertEqual(session.state, LiveState.LISTENING)
        snapshot = controller.ngrams.snapshot()
        self.assertEqual(
            snapshot["frames_seen"], 5 + controller.segmenter.start_speech_frames
        )
        self.assertGreater(snapshot["unique_ngrams"][2], 0)
        self.assertGreater(snapshot["unique_ngrams"][3], 0)
        self.assertGreater(snapshot["unique_ngrams"][4], 0)

    def test_vad_segmenter_preserves_preroll_and_uses_hysteresis(self):
        config = KiraLiveConfig().audio
        segmenter = VadSegmenter(config)
        frame = [0.0] * config.samples_per_frame
        for _ in range(segmenter.pre_roll_frames):
            self.assertEqual(segmenter.process(frame, 0.1), ())
        for _ in range(segmenter.start_speech_frames - 1):
            self.assertEqual(segmenter.process(frame, 0.9), ())
        started = segmenter.process(frame, 0.9)
        self.assertEqual(started[0].kind, "speech_start")
        for _ in range(segmenter.end_silence_frames - 1):
            self.assertEqual(segmenter.process(frame, 0.1), ())
        ended = segmenter.process(frame, 0.1)
        self.assertEqual(ended[0].kind, "speech_end")
        self.assertGreaterEqual(
            len(ended[0].samples),
            (segmenter.pre_roll_frames + segmenter.end_silence_frames) * config.samples_per_frame,
        )

    def test_qwen4_micro_blueprint_maps_all_backbone_layers(self):
        blueprint = Qwen4MicroBlueprint.kira_live_1()
        blueprint.validate()
        self.assertEqual(len(blueprint.layers), 24)
        self.assertEqual(
            [layer.index for layer in blueprint.layers if layer.token_mixer == "qwen_sparse_attention"],
            [4, 8, 12, 16, 20, 24],
        )
        self.assertEqual(sum(layer.ple_ngram for layer in blueprint.layers), 6)
        self.assertEqual(sum(layer.expert_moe for layer in blueprint.layers), 16)
        self.assertLess(blueprint.added_parameter_estimate()["total"], 200_000_000)
        self.assertEqual(blueprint.ple_ngram_sizes, (2, 3, 4))
        self.assertEqual(len(blueprint.expert_names), 10)
        self.assertEqual(blueprint.expert_rank, 128)
        self.assertLess(
            blueprint.total_deployment_parameter_estimate()["total"],
            4_000_000_000,
        )

    def test_composite_is_one_graph_with_three_weight_islands_and_no_prompt_handoff(self):
        plan = KiraLiveCompositePlan.kira_live_1()
        plan.validate()
        self.assertEqual(
            (plan.listener.role, plan.thinker.role, plan.talker.role),
            ("listener", "thinker", "talker"),
        )
        self.assertFalse(plan.text_prompt_handoff)
        self.assertTrue(plan.transcript_is_auxiliary)
        self.assertEqual(plan.ngram_domains, ("listener_units", "thinker_tokens", "talker_codes"))
        self.assertEqual(plan.expert_top_k, 2)
        self.assertEqual(plan.listener.hidden_width, 1024)
        self.assertEqual(plan.listener_to_thinker.input_width, 1024)
        self.assertLess(plan.deployment_parameter_estimate()["total"], 4_000_000_000)

    def test_manifest_scopes_moe_to_thinker_and_requires_true_barge_in(self):
        import json

        manifest = json.loads(
            (Path(__file__).parents[1] / "kira_live" / "model_manifest.json").read_text()
        )
        composition = manifest["composition"]
        self.assertEqual(composition["moe_scope"], "thinker_only")
        self.assertIn("no_moe", composition["listener_routing"])
        self.assertIn("no_moe", composition["talker_routing"])
        self.assertEqual(
            composition["barge_in"],
            "speech_onset_cancels_thinker_tokens_and_talker_playback",
        )

    def test_pinned_composite_weights_resolve_from_local_cache(self):
        weights = resolve_composite_weights()
        self.assertEqual(
            tuple(weight.spec.role for weight in weights.by_role()),
            ("listener", "thinker", "talker"),
        )
        for weight in weights.by_role():
            self.assertTrue(weight.model_file.is_file())
        self.assertEqual(
            tuple(weight.spec.hidden_width for weight in weights.by_role()),
            (1024, 1024, 2048),
        )
        self.assertIn("1.7B-CustomVoice", weights.talker.spec.model_id)

    def test_logical_checkpoint_bundle_and_parameter_budget_match(self):
        import json

        bundle = resolve_live_checkpoint()
        self.assertTrue(bundle.thinker_addons.is_file())
        self.assertTrue(bundle.emotion_head.is_file())
        self.assertEqual(bundle.weights.talker.spec.hidden_width, 2048)
        manifest = json.loads(
            (Path(__file__).parents[1] / "kira_live" / "model_manifest.json").read_text()
        )
        budget = manifest["parameter_budget"]
        components = (
            "listener", "thinker", "thinker_addons", "talker",
            "talker_codec", "emotion_head",
        )
        self.assertEqual(sum(budget[name] for name in components), budget["estimated_total"])
        self.assertEqual(
            budget["hard_cap"] - budget["estimated_total"],
            budget["estimated_headroom"],
        )

    def test_private_think_block_is_removed_before_ui_memory_or_speech(self):
        value = sanitize_public_output(
            "<think>I should never reveal this plan.</think>Hello, I can help."
        )
        self.assertEqual(value, "Hello, I can help.")

    def test_integrated_voice_uses_thinker_tokens_and_direct_emotion_state(self):
        class Tokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                self.received = (token_ids, skip_special_tokens)
                return "I understand. Let us fix it."

        tokenizer = Tokenizer()
        emotion = EmotionState(
            {"neutral": 0.05, "joy": 0.05, "sadness": 0.8, "anger": 0.02, "fear": 0.05, "surprise": 0.03},
            valence=-0.6,
            arousal=0.2,
            confidence=0.9,
            observed_at=0.0,
        )
        plan = KiraIntegratedQwenVoice.build_plan(
            (11, 12, 13),
            "I understand. Let us fix it.",
            tokenizer,
            emotion,
        )
        self.assertEqual(tokenizer.received, ([11, 12, 13], True))
        self.assertEqual(plan.token_ids, (11, 12, 13))
        self.assertIn("gentle", plan.style_instruction)
        self.assertIn("unhurried", plan.style_instruction)
        self.assertNotIn("sadness", plan.public_text)

    def test_uncertain_emotion_keeps_talker_neutral(self):
        state = EmotionState(
            {"neutral": 0.2, "joy": 0.8, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0},
            valence=0.8,
            arousal=0.9,
            confidence=0.2,
            observed_at=0.0,
        )
        instruction = emotion_style_instruction(state)
        self.assertIn("steadily", instruction)
        self.assertIn("Do not exaggerate", instruction)

    def test_live_readiness_reports_verified_native_graph(self):
        status = kira_live_readiness()
        self.assertTrue(status["ui_ready"])
        self.assertTrue(status["conversation_ready"])
        self.assertFalse(status["prompt_handoffs"])
        self.assertFalse(status["private_reasoning_visible"])
        checks = {check["key"]: check for check in status["checks"]}
        self.assertTrue(checks["listener"]["ready"])
        self.assertTrue(checks["thinker"]["ready"])
        self.assertTrue(checks["codec_groups"]["ready"])
        self.assertTrue(checks["waveform"]["ready"])
        self.assertNotIn("Streaming Listener hidden states", status["blockers"])
        self.assertNotIn("All 16 codec groups", status["blockers"])
        self.assertNotIn("Speech-tokenizer waveform decoder", status["blockers"])
        checks = {check["key"]: check for check in status["checks"]}
        self.assertTrue(checks["semantic_bridge"]["ready"])
        self.assertTrue(checks["thinker_addons"]["ready"])
        self.assertTrue(checks["talker_alignment"]["ready"])
        self.assertTrue(checks["session"]["ready"])
        self.assertEqual(status["blockers"], [])

    def test_composite_runtime_never_passes_listener_transcript_to_another_island(self):
        calls = []

        class Listener:
            def encode(self, audio_samples, sample_rate):
                calls.append(("listener", audio_samples, sample_rate))
                return ListenerState("listener-hidden", "listener-units", "SECRET TRANSCRIPT")

        class Thinker:
            def seed(self):
                calls.append(("thinker-seed",))
                return ThinkerSeed("seed-hidden", "seed-tokens", "memory")

            def decode_hidden(self, fused_hidden, token_ids):
                calls.append(("thinker-hidden", fused_hidden, token_ids))
                return ThinkerState("thought-hidden", "thought-tokens")

        class Bridge:
            def fuse(self, **kwargs):
                self.received = kwargs
                calls.append(("bridge", kwargs["listener_hidden"]))
                return CompositeBridgeState("fused", "semantic", "codec", "routes")

        class Talker:
            def synthesize_hidden(self, **kwargs):
                self.received = kwargs
                calls.append(("talker", kwargs["semantic_conditioning"]))
                return "audio", 24_000

        bridge = Bridge()
        talker = Talker()
        runtime = KiraLiveCompositeRuntime(Listener(), Thinker(), bridge, talker)
        output = runtime.respond(
            "samples",
            sample_rate=16_000,
            emotion_features="emotion-vector",
            talker_code_ids="codec-prefix",
        )
        self.assertEqual(output.auxiliary_transcript, "SECRET TRANSCRIPT")
        self.assertNotIn("SECRET TRANSCRIPT", repr(bridge.received))
        self.assertNotIn("SECRET TRANSCRIPT", repr(talker.received))
        self.assertEqual(
            [call[0] for call in calls],
            ["listener", "thinker-seed", "bridge", "thinker-hidden", "talker"],
        )

    def test_persistent_workflow_memory_keeps_all_events_and_retrieves_old_facts(self):
        with tempfile.TemporaryDirectory() as root:
            store = PersistentWorkflowMemory(Path(root) / "memory.sqlite3")
            store.append("chat", "user", "The launch codename is violet glacier.")
            for index in range(40):
                store.append("chat", "assistant", f"Completed routine step {index}.")
            store.upsert_workflow_item(
                "release",
                "verify",
                "blocked",
                "Verify the signed package.",
                dependencies=["build"],
            )
            context = store.build_context(
                "chat",
                "What was the launch codename?",
                workflow_id="release",
                recent=4,
            )
            self.assertEqual(store.count("chat"), 41)
            self.assertIn("violet glacier", context)
            self.assertIn("Verify the signed package", context)

    def test_typed_chat_and_live_voice_share_one_idempotent_memory(self):
        with tempfile.TemporaryDirectory() as root:
            store = PersistentWorkflowMemory(Path(root) / "memory.sqlite3")
            messages = [
                {"id": "typed-1", "role": "user", "content": "My dog's name is Pixel.", "created_at": 1.0},
                {"id": "voice-1", "role": "assistant", "content": "I remember Pixel.", "memory_synced": True},
            ]
            store.append("chat", "assistant", "I remember Pixel.")
            self.assertEqual(store.sync_chat_messages("chat", messages), 1)
            self.assertEqual(store.sync_chat_messages("chat", messages), 0)
            self.assertEqual(store.count("chat"), 2)
            context = store.build_context("chat", "What is my dog's name?")
            self.assertIn("Pixel", context)
            self.assertEqual(
                grounded_recall("What is my dog's name?", context),
                "You told me your dog's name is Pixel.",
            )
            self.assertIsNone(grounded_recall("What is my cat's name?", context))

    def test_persistent_workflow_memory_redacts_secrets_without_dropping_turn(self):
        with tempfile.TemporaryDirectory() as root:
            store = PersistentWorkflowMemory(Path(root) / "memory.sqlite3")
            store.append("chat", "user", "api_key=sk-example-super-secret-value")
            events = list(store.events("chat"))
            self.assertEqual(len(events), 1)
            self.assertIn("sensitive content omitted", events[0].content)
            self.assertNotIn("sk-example", events[0].content)

    def test_memory_budget_keeps_header_and_complete_recent_events(self):
        with tempfile.TemporaryDirectory() as root:
            store = PersistentWorkflowMemory(Path(root) / "memory.sqlite3")
            for index in range(8):
                store.append("chat", "user", f"complete-memory-event-{index}")
            context = store.build_context("chat", "memory", recent=8, max_chars=230)
            self.assertTrue(context.startswith("PERSISTENT CHAT MEMORY"))
            self.assertIn("complete-memory-event-7", context)
            self.assertLessEqual(len(context), 230)
            self.assertTrue(
                all(
                    not line.startswith("- ") or "complete-memory-event-" in line
                    for line in context.splitlines()
                )
            )

    def test_live_session_writes_and_retrieves_persistent_chat_memory(self):
        with tempfile.TemporaryDirectory() as root:
            memory = PersistentWorkflowMemory(Path(root) / "memory.sqlite3")
            session = KiraLiveSession(memory=memory, chat_id="chat", workflow_id="build")
            session.start()
            event = session.transcript_ready("Remember the release color is indigo.")
            self.assertIn("release color is indigo", event.payload["memory_context"])
            session.response_finished("I recorded the release color.")
            self.assertEqual(memory.count("chat"), 2)


if __name__ == "__main__":
    unittest.main()
