import unittest
from orchestrator_v15.horizon_context import ContextSpan as S, expand_layout, generalized_attention_mask


def layout():
    return [S('text',2,0),S('semantic_vision',2,1,image_id='source'),
            S('clean_latent',2,2,image_id='source'),S('relation',2,3),
            S('noisy_latent',2,4,image_id='target'),S('clean_latent',2,5,image_id='target'),
            S('text',2,6),S('text',1,0,sample=1)]


class SharedContextMaskTests(unittest.TestCase):
    def test_text_causal_visual_bidirectional_and_no_future(self):
        mask=generalized_attention_mask(layout())
        self.assertFalse(mask[0][1])
        self.assertTrue(mask[1][0])
        self.assertTrue(mask[2][3])
        self.assertTrue(mask[8][9])
        self.assertFalse(mask[8][10])
        self.assertTrue(mask[8][4])
        self.assertTrue(mask[8][6])

    def test_later_context_uses_clean_not_noisy_images(self):
        mask=generalized_attention_mask(layout())
        self.assertTrue(mask[12][10])
        self.assertTrue(mask[12][2])
        self.assertFalse(mask[12][8])
        self.assertFalse(mask[10][8])

    def test_packed_samples_are_isolated(self):
        mask=generalized_attention_mask(layout())
        for index in range(14):
            self.assertFalse(mask[index][14])
            self.assertFalse(mask[14][index])

    def test_target_cannot_read_earlier_clean_target_or_target_semantics(self):
        spans=[S('text',1,0),S('semantic_vision',1,1,image_id='target'),
               S('clean_latent',1,2,image_id='target'),S('noisy_latent',1,3,image_id='target')]
        mask=generalized_attention_mask(spans)
        self.assertFalse(mask[3][1]); self.assertFalse(mask[3][2]); self.assertTrue(mask[3][3])

    def test_invalid_layout_and_reference_limit_rejected(self):
        for spans in ([S('noisy_latent',1,0)], [S('text',1,0),S('relation',1,0)],
                      [S('text',0,0)], [S('unknown',1,0)]):
            with self.assertRaises(ValueError):
                expand_layout(spans)
        with self.assertRaises(ValueError):
            expand_layout([S('text',10,0)],max_tokens=9)
