import unittest
from main import apply_lora_all_params, freeze_non_lora_params, enable_disable_lora_all_params
from utils import get_device, BigClassifier
import numpy as np

class TestLoRALinearLayer(unittest.TestCase):
    def setUp(self):
        self.device = get_device()
        self.big_classifier = BigClassifier(28 * 28, 10).to(self.device)  # for MNIST

    def test_correct_params(self):
        total_params_non_lora = 0
        for name, param in self.big_classifier.named_parameters():
            total_params_non_lora += np.prod(param.shape)

        expected_total_params_non_lora = 28 * 28 * 256 + 256 + 256 * 64 + 64 + 64 * 10 + 10  # weights and biases
        expected_total_params_lora = 28 * 28 + 256 + 256 + 64 + 64 + 10  # each a * b translates into a + b because the rank is 1
        self.assertEqual(total_params_non_lora, expected_total_params_non_lora)

        apply_lora_all_params(self.big_classifier, self.device)

        total_params_non_lora = 0
        total_params_lora = 0

        for name, param in self.big_classifier.named_parameters():
            if "lora" in name:
                total_params_lora += np.prod(param.shape)
            elif "original" in name or "bias" in name:
                total_params_non_lora += np.prod(param.shape)
            else:
                raise AssertionError("Unexpected type of params")

        self.assertEqual(total_params_non_lora, expected_total_params_non_lora)
        self.assertEqual(total_params_lora, expected_total_params_lora)

        freeze_non_lora_params(self.big_classifier)

        for name, param in self.big_classifier.named_parameters():
            if "lora" in name:
                self.assertTrue(param.requires_grad)
            else:
                self.assertFalse(param.requires_grad)



if __name__ == '__main__':
    unittest.main()
