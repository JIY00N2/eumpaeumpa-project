import logging

import string
import soundfile
from .VAD import VoiceActivityDecection

import numpy as np
from espnet2.bin.asr_inference import Speech2Text

class SpeechRecognitionModel(object):
    def __init__(self, config, *args, **kwargs):
        self.speech2text = Speech2Text(
            **config,
            maxlenratio=0.0,
            minlenratio=0.0,
            beam_size=20,
            ctc_weight=0.3,
            penalty=0.0,
            nbest=1)

        self.vad = VoiceActivityDecection()


    def __call__(self, audio_path, *args, **kwargs):
        vad_result = self.vad(audio_path)
        hypothesis = self.asr_proc(audio_path, vad_result)

        return hypothesis

    def text_normalizer(self, text):
        text = text.upper()
        return text.translate(str.maketrans('', '', string.punctuation))


    def asr_proc(self, audio_path, vad_result):
        audio, fs = soundfile.read(audio_path)

        fsh10 = int(fs / 100)
        is_speech = 1
        start = 0
        end = 0

        hypothesis = []
        for (idx, is_active) in enumerate(vad_result):
            if is_speech == is_active:
                if is_speech == 1:
                    start = idx
                    is_speech = 1 - is_speech

                elif is_speech == 0:
                    end = idx
                    is_speech = 1 - is_speech
                    if end - start < 13:
                        continue

                    start_idx = int(start * fsh10)
                    end_idx = int(end*fsh10)

                    nbest = self.speech2text(audio[start_idx:end_idx])
                    text, *_ = nbest[0]
                    hypothesis.append(self.text_normalizer(text))
                    #print("segmented: ", start * 0.01, "s ~", end * 0.01, "s")
                    #print(f"ASR hypothesis: {self.text_normalizer(text)}")
                    #display(Audio(audio[int(start * fsh10):int(end * fsh10)], rate=rate))
                    logging.info("segmented: ", start * 0.01, "s ~", end * 0.01, "s")
                    logging.info(self.text_normalizer(text))
        return hypothesis

