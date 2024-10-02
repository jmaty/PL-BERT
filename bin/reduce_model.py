#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import argparse
import yaml
import torch
from transformers import AlbertConfig, AlbertModel


class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state


def load_plbert(model_file, config_file):
    plbert_config = yaml.safe_load(open(config_file, encoding='utf-8'))

    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    bert = CustomAlbert(albert_base_configuration)

    checkpoint = torch.load(model_file, map_location='cpu')
    state_dict = checkpoint['net']
    step = checkpoint['step']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        if name.startswith('encoder.'):
            name = name[8:] # remove `encoder.`
            new_state_dict[name] = v
    if "embeddings.position_ids" in new_state_dict:
        del new_state_dict["embeddings.position_ids"]
    bert.load_state_dict(new_state_dict, strict=False)
    return bert, step


def save_model(model, path, step=None):
    state_dict = {
        'net': model.state_dict()
    }
    if step:
        state_dict['step'] = step
    torch.save(state_dict, path)


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Reduce model for use in StyleTTS2.\n\n
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "inp_model",
        type=str,
        default=None,
        help="Input model path.")
    parser.add_argument(
        "out_model",
        type=str,
        default=None,
        help="Output model path.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        required=True,
        help="Model config")
    args = parser.parse_args()

    model, step = load_plbert(args.inp_model, args.config)
    save_model(model, args.out_model, step+1)
    print(f'Model saved to {args.out_model} with step {step+1}')

if __name__ == "__main__":
    main()
