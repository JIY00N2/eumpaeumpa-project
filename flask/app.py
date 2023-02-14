import os, sys
from subprocess import call
from flask import Flask, render_template, request, redirect, url_for, flash
from ASRmodules import SpeechRecognitionModel
from werkzeug.utils import secure_filename
from tts import tts_model
import torch
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from gcp.bucket import Bucket_processor
import argparse

app = Flask(__name__)

path = os.getcwd()
main_path = path[:-6]

asr_model_file = main_path + '\\data\\asr_config\\asr_train_asr_conformer_raw_kr_bpe2309\\31epoch.pth'
asr_train_config = main_path + '\\data\\asr_config\\asr_train_asr_conformer_raw_kr_bpe2309\\config.yaml'
config = {
        'asr_model_file': asr_model_file,
        'asr_train_config':asr_train_config
}

asr = SpeechRecognitionModel(config)
audio_dir = main_path + "\\flask\\templates\\result_dir\\"

audio_wav_path = os.path.join(audio_dir, "audio_wav.wav")

#sst
@app.route('/', methods=['GET', 'POST'])
def stthtml():
        if request.method == 'GET':
                return render_template('stt.html')
        elif request.method == 'POST':
                # get input values
                if 'file' not in request.files:
                        flash('No file part')
                        return redirect(request.url)
                f = request.files['file']
                audio_path = audio_dir + secure_filename(f.filename)
                f.save(audio_path)
                call('ffmpeg -y -i {} -ar 16000 -ac 1 {}'.format(audio_path, audio_wav_path), shell=True)

                hyps = asr(audio_wav_path)
                result = " \n".join(hyps)
                return render_template('stt.html', asr_result=result)

#tts
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
KEY_PATH = ABS_PATH + "/../gcp_auth_key/capstone-352301-ef799c59a451.json"
SAVE_PATH = ABS_PATH + "/models/epoch 10 weight 20000.pt"
bucket_processor = Bucket_processor(KEY_PATH, "capstone-352301", "capstone_mlops_data")
# gcs에 data가 있다고 가정함
bucket_processor.download_from_bucket("capstone_data/best_model/epoch 10 weight 20000.pt", SAVE_PATH)

device = torch.device("cpu")
ttsm = tts_model('ainize/kobart-news', './models/epoch 10 weight 20000.pt', device)

@app.route('/tts',methods=['POST','GET'])
def ttshtml():
    if request.method == 'GET':
        return render_template('tts.html')

    if request.method == 'POST':
        text = request.form['content']
        print(text)
        text = ttsm.preprocessing(text)
        sum_text = ttsm.get_result(text)
        print(sum_text)

        return render_template('tts.html', summary_result=sum_text, content_value=text)
#서버실행
if __name__ == '__main__':
        app.run(debug=True)

