import unittest

from kira_live.token_bus import remap_token_ids


class Source:
    def __init__(self, pieces): self.pieces = pieces
    def convert_ids_to_tokens(self, ids): return self.pieces


class Target:
    unk_token_id = None
    def __init__(self, vocab): self.vocab = vocab
    def convert_tokens_to_ids(self, pieces):
        if isinstance(pieces, str): return self.vocab.get(pieces, self.unk_token_id)
        return [self.vocab.get(piece, self.unk_token_id) for piece in pieces]


class TokenBusTests(unittest.TestCase):
    def test_known_pieces_keep_the_fast_path(self):
        self.assertEqual(remap_token_ids([1,2], Source(['Hey','ĠKira']), Target({'Hey':4,'ĠKira':8})), (4,8))

    def test_missing_merge_splits_without_decoding_utf8_fragments(self):
        # The exact partial-byte merge that failed for the greeting हे किरा.
        self.assertEqual(remap_token_ids([1], Source(['à¤¿à¤']), Target({'à¤¿':3,'à¤':7})), (3,7))

    def test_unknown_sentinel_also_uses_lossless_piece_splitting(self):
        target=Target({'ab':1,'c':2}); target.unk_token_id=99
        self.assertEqual(remap_token_ids([1,2],Source(['abc','abc']),target),(1,2,1,2))

    def test_unrepresentable_piece_has_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'cannot be represented'):
            remap_token_ids([1],Source(['xyz']),Target({'x':1}))

    def test_missing_control_token_is_not_spelled_out(self):
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            remap_token_ids([1],Source(['<|missing|>']),Target({'<':1}))

    def test_real_qwen_greeting_and_multilingual_roundtrip(self):
        from transformers import AutoTokenizer
        from kira_live.weights import resolve_composite_weights
        weights=resolve_composite_weights()
        source=AutoTokenizer.from_pretrained(str(weights.listener.snapshot),local_files_only=True)
        target=AutoTokenizer.from_pretrained(str(weights.thinker.snapshot),local_files_only=True)
        for text in ['Hey Kira.','हे किरा','नमस्ते किरा, आप कैसे हैं?','こんにちは、キラ','你好，Kira','مرحبا كيرا','🙂 café']:
            with self.subTest(text=text):
                ids=source.encode(text,add_special_tokens=False)
                mapped=remap_token_ids(ids,source,target)
                self.assertEqual(target.decode(list(mapped)),text)
