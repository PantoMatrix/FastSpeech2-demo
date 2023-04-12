import re
import argparse
from string import punctuation

import torch
import yaml
import json
import numpy as np
from g2p_en import G2p

from text import text_to_sequence
from model import FastSpeech2

import hifigan


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def preprocess_english(text):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon("librispeech-lexicon.txt")
    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")
    sequence = np.array(text_to_sequence(phones, ["english_cleaners"]))
    return np.array(sequence)

def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


class FS2:
    def __init__(self, model_name='en_16', device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(FS2, self).__init__()
        self.preprocess_config = yaml.load(open(f'config/{model_name}/preprocess.yaml'), Loader=yaml.FullLoader)
        self.model_config = yaml.load(open(f'config/{model_name}/model.yaml'), Loader=yaml.FullLoader)
        with open("speakers.json", "r") as f:
            self.speakers_id = json.load(f)
        self.device = device
        self.model_init()
        self.vocoder_init()
        
    def model_init(self):
        self.model = FastSpeech2(self.preprocess_config, self.model_config).to(self.device)
        ckpt = torch.load("150000.pth.tar")
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()
        self.model.requires_grad_ = False
        
    def vocoder_init(self):
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        self.vocoder = hifigan.Generator(config)
        ckpt = torch.load("generator_universal.pth.tar")
        self.vocoder.load_state_dict(ckpt["generator"])
        self.vocoder.eval()
        self.vocoder.remove_weight_norm()
        self.vocoder.to(self.device)
        
    def __call__(self, text, speaker):
        phonemes = np.array(([preprocess_english(text)]))
        speaker = np.array([self.speakers_id[str(speaker)]])
        src_len = np.array([len(phonemes[0])])
        
        predictions =  self.model(torch.from_numpy(speaker).long().to(self.device), 
                                  torch.from_numpy(phonemes).long().to(self.device), 
                                  torch.from_numpy(src_len).long().to(self.device),
                                  max(src_len))
        
        src_len = predictions[8][0].item()
        mel_len = predictions[9][0].item()
        mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
        duration = predictions[5][0, :src_len].detach().cpu().numpy()
        pitch = predictions[2][0, :src_len].detach().cpu().numpy()
        pitch = expand(pitch, duration)
        energy = predictions[3][0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
        
        with torch.no_grad():
            wavs = self.vocoder(mel_prediction.unsqueeze(0)).squeeze(1)
        return wavs
