import unittest

import torch.optim
from torch import nn
from torchvision import datasets
from torchvision.transforms import transforms

import utils
from main import apply_lora_all_params, freeze_non_lora_params, enable_disable_lora_all_params, apply_lora_single_layer
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

    def test_correct_weights_after_fine_tuning(self):
        # simulating a pretrained model
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_dataloader = torch.utils.data.DataLoader(mnist_train, batch_size=16, shuffle=True)

        cross_entropy = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.big_classifier.parameters(), lr = 1e-3)
        utils.train(self.big_classifier, train_dataloader, optimizer, cross_entropy, 0, self.device)

        original_weights = {}
        for name, param in self.big_classifier.named_parameters():
            original_weights[name] = param.clone().detach()

        # applying LoRA
        apply_lora_all_params(self.big_classifier, self.device)
        freeze_non_lora_params(self.big_classifier)

        # Fine Tuning
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        exclude_indices = mnist_train.targets == 7
        mnist_train.data = mnist_train.data[exclude_indices]
        mnist_train.targets = mnist_train.targets[exclude_indices]

        train_dataloader = torch.utils.data.DataLoader(mnist_train, batch_size=16, shuffle=True)
        optimizer = torch.optim.Adam(self.big_classifier.parameters(), lr=1e-3)
        utils.train(self.big_classifier, train_dataloader, optimizer, cross_entropy, 0, self.device)

        # Assertions
        for name, param in self.big_classifier.named_parameters():
            if 'original' in name:
                name = name.replace('.original', '')
                name = name.replace('.parametrizations', '')
                self.assertEqual(torch.all(param - original_weights[name]), 0)

        # enable lora inference
        enable_disable_lora_all_params(self.big_classifier, True)
        self.assertTrue(
            torch.equal(
                self.big_classifier.sequential[1].weight,
                self.big_classifier.sequential[1].parametrizations.weight.original + (self.big_classifier.sequential[1].parametrizations.weight[0].lora_B @ self.big_classifier.sequential[1].parametrizations.weight[0].lora_A)
            )
        )

        # disable lora inference
        enable_disable_lora_all_params(self.big_classifier, enabled=False)
        self.assertTrue(torch.equal(self.big_classifier.sequential[1].weight, original_weights['sequential.1.weight']) )
        self.assertTrue(torch.equal(self.big_classifier.sequential[3].weight, original_weights['sequential.3.weight']))
        self.assertTrue(torch.equal(self.big_classifier.sequential[5].weight, original_weights['sequential.5.weight']))

class TestLoRAConv2dLayer(unittest.TestCase):
    def setUp(self):
        self.device = get_device()


    def test_simple_conv_layer(self):
        conv_layer = torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(2, 7), device=self.device)
        # sample_input = torch.rand(size=(3, 32, 32), device=self.device)
        total_params_non_lora = 0
        for name, param in conv_layer.named_parameters():
            total_params_non_lora += np.prod(param.shape)

        expected_total_params_non_lora = 3*5*2*7 + 5
        expected_total_params_lora = 3*2 + 7*5
        self.assertEqual(total_params_non_lora, expected_total_params_non_lora)

        apply_lora_all_params(conv_layer, self.device)

        total_params_non_lora = 0
        total_params_lora = 0

        for name, param in conv_layer.named_parameters():
            if "lora" in name:
                total_params_lora += np.prod(param.shape)
            elif "original" in name or "bias" in name:
                total_params_non_lora += np.prod(param.shape)
            else:
                raise AssertionError("Unexpected type of params")

        self.assertEqual(total_params_non_lora, expected_total_params_non_lora)
        self.assertEqual(total_params_lora, expected_total_params_lora)

        freeze_non_lora_params(conv_layer)

        for name, param in conv_layer.named_parameters():
            if "lora" in name:
                self.assertTrue(param.requires_grad)
            else:
                self.assertFalse(param.requires_grad)

    def test_full_cnn_classifier_cifar10(self):
        # TODO: implement
        pass



if __name__ == '__main__':
    unittest.main()
