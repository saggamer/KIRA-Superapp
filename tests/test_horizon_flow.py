import unittest
import torch
from orchestrator_v15.horizon_flow_torch import sample_flow_timesteps,flow_training_pair
class FlowTests(unittest.TestCase):
 def test_seeded_logit_normal_and_shift(self):
  a=sample_flow_timesteps(128,generator=torch.Generator().manual_seed(7))
  b=sample_flow_timesteps(128,generator=torch.Generator().manual_seed(7))
  c=sample_flow_timesteps(128,generator=torch.Generator().manual_seed(7),shift=2)
  self.assertTrue(torch.equal(a,b));self.assertTrue(((a>0)&(a<1)).all());self.assertTrue((c>a).all())
 def test_velocity_sign_and_endpoints(self):
  clean=torch.randn(2,3,4);noise=torch.randn_like(clean);t=torch.tensor([0.,1.])
  noised,velocity=flow_training_pair(clean,noise,t)
  torch.testing.assert_close(noised[0],clean[0]);torch.testing.assert_close(noised[1],noise[1])
  torch.testing.assert_close(noised[1]-velocity[1],clean[1])
  self.assertFalse(velocity.requires_grad)
 def test_invalid_time_and_shapes(self):
  for shift in [0,-1,float('nan')]:
   with self.assertRaises(ValueError):sample_flow_timesteps(1,shift=shift)
  with self.assertRaises(ValueError):flow_training_pair(torch.zeros(1,3),torch.zeros(2,3),torch.zeros(1))
  with self.assertRaises(ValueError):flow_training_pair(torch.zeros(1,3),torch.zeros(1,3),torch.tensor([2.]))
