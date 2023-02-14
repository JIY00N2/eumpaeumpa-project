import os

from ASRmodules import SpeechRecognitionModel


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = os.getcwd()
    config = {
        'asr_model_file':'C:\\Users\\shxod\\PycharmProjects\\Capstone\\data\\asr_config\\asr_train_asr_conformer_raw_kr_bpe2309\\31epoch.pth',
        'asr_train_config':'C:\\Users\shxod\\PycharmProjects\\Capstone\\data\\asr_config\\asr_train_asr_conformer_raw_kr_bpe2309\\config.yaml'
    }
    audio_path = 'C:\\Users\\shxod\\PycharmProjects\\Capstone\\data\\soundfile.wav'
    asr = SpeechRecognitionModel(config)
    hyps = asr(audio_path)

    print()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
