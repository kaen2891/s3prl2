from glob import glob
import os, shutil
import soundfile as sf
import argparse
import numpy as np
import librosa
import soundfile as sf

parser = argparse.ArgumentParser(description='copy')
parser.add_argument('--start_dir', type=str, default='/NasData/home/junewoo/workspace/asr2/script/')
parser.add_argument('--dest_dir', type=str, default='/Data~')
args = parser.parse_args()



def move1(dataset, data_dir):
    for data in dataset:
        _, file_id = os.path.split(data)
        print(file_id)
        audio_extension = file_id.split('.')[-1].lower()
        assert audio_extension in ('wav', 'mp3', 'flac', 'pcm'), f"Unsupported format: {audio_extension}"
        
        if audio_extension == 'pcm':
            with open(data, 'rb') as opened_pcm_file:
                buf = opened_pcm_file.read()
                pcm_data = np.frombuffer(buf, dtype = 'int16')
                wav_data = librosa.util.buf_to_float(pcm_data, 2)
            sf.write(os.path.join(data_dir, file_id.replace('.pcm', '.wav')), wav_data, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
            
        else:
            based_sr = librosa.get_samplerate(input_file)
            if based_sr != 16000:
                y, _ = librosa.load(y=input_file, sr=based_sr)
                input_file = resampling(y, based_sr, 16000)
                sf.write(os.path.join(data_dir, file_id), input_file, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
            else:
                shutil.copy(data, data_dir)
    

def move2(dataset, data_dir):
    for data in dataset:
        shutil.copy(data, data_dir)

def move3(data, data_dir):
    shutil.copy(data, data_dir)


train_path = os.path.join(args.start_dir, 'KoSpeech_1000hour')
valid_path = os.path.join(args.start_dir, 'valid_1000hour')

target_train_path = os.path.join(args.dest_dir, 'KoSpeech_1000hour')
if not os.path.exists(target_train_path):
    os.makedirs(target_train_path)
target_valid_path = os.path.join(args.dest_dir, 'valid_1000hour')
if not os.path.exists(target_valid_path):
    os.makedirs(target_valid_path)

#s_t_speech = sorted(glob(train_path+'/*.wav'))

s_t_script = sorted(glob(train_path+'/*.script'))

#s_v_speech = sorted(glob(valid_path+'/*.wav'))

s_v_script = sorted(glob(valid_path+'/*.script'))

#print('train script {} train label {}'.format(len(s_t_script), len(s_t_label)))
#print('valid script {} valid label {}'.format(len(s_v_script), len(s_v_label)))
#print('train pcm {} valid pcm {}'.format(len(s_t_speech), len(s_v_speech)))

#move1(s_t_speech, target_train_path)
#move2(s_t_speech, target_train_path)

#print('Extracting audio length...', flush=True)
#tr_x = Parallel(n_jobs=args.-1)(delayed(move1)(s_t_speech, target_train_path) for file in tqdm(todo))

#move1(s_v_speech, target_valid_path)
#move2(s_v_speech, target_valid_path)

#exit()

move2(s_t_script, target_train_path)
move2(s_v_script, target_valid_path)




train_csv = os.path.join(train_path, 'data_list.csv')
train_label = os.path.join(train_path, 'train_label')
valid_csv = os.path.join(valid_path, 'data_list.csv')
valid_label = os.path.join(valid_path, 'valid_label')


move3(train_csv, target_train_path)
move3(train_label, target_train_path)

move3(valid_csv, target_valid_path)
move3(valid_label, target_valid_path)
