import unittest
from more.quad_model import *

class TestQuadModel(unittest.TestCase):
    def test_ema(self):

        model_options_rls = {"whiten_input": False,
                             "normalize_features": True,
                             "normalize_output": False,
                             "unnormalize_output": False,
                             "output_weighting": False,
                             "alpha" : 0.2,
                             "delta": 100}

        rls_model = QuadModelRLS(5, model_options_rls)
        print("ema recursive")
        print(rls_model.ema_step(np.array([1,1,1,1,1])))
        print(rls_model.ema_step(np.array([2,1,1,1,1])))
        print(rls_model.ema_step(np.array([1,1,1,1,1])))
        print(rls_model.ema_step(np.array([1,1,1,1,1])))
        print(rls_model.ema_step(np.array([1,1,1,1,1])))
        print(rls_model.ema_step(np.array([1,1,1,1,1])))

    def test_ema_with_sum_count(self):
        model_options_rls = {"whiten_input": False,
                             "normalize_features": True,
                             "normalize_output": False,
                             "unnormalize_output": False,
                             "output_weighting": False,
                             "alpha" : 0.2,
                             "delta": 100}

        rls_model = QuadModelRLS(5, model_options_rls)
        print("\nema with sum and count")
        print(rls_model.ema_with_sum_count(np.array([1,1,1,1,1])))
        print(rls_model.ema_with_sum_count(np.array([2,1,1,1,1])))
        print(rls_model.ema_with_sum_count(np.array([1,1,1,1,1])))
        print(rls_model.ema_with_sum_count(np.array([1,1,1,1,1])))
        print(rls_model.ema_with_sum_count(np.array([1,1,1,1,1])))
        print(rls_model.ema_with_sum_count(np.array([1,1,1,1,1])))

if __name__ == '__main__':
    unittest.main()
