# python=3.10
# datasets, transformers, torch, librosa, soundfile, fastapi, pydub, torchaudio, python-multipart
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

import tempfile
from pydub import AudioSegment
import torchaudio
import soundfile as sf
import io

model_id = "facebook/mms-1b-all"

processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

processor.tokenizer.set_target_lang("crk-script_latin")
model.load_adapter("crk-script_latin")

app = FastAPI()

def my_inference_function(audio_file):
        
    inputs = processor(audio_file, sampling_rate=16_000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)

    print(transcription)
    return(transcription)


def convert_blob_to_wav(binary_blob, output_file):
    # print("START")

    if binary_blob.startswith(b'RIFF'):
        audio = AudioSegment.from_file(io.BytesIO(binary_blob), format="wav")

    # Check if the binary blob starts with the "EBML" header, indicating WebM format
    if binary_blob.startswith(b'\x1a\x45\xdf\xa3'):
        audio = AudioSegment.from_file(io.BytesIO(binary_blob), format="webm")
    
    # Save the audio as a WAV file
    audio.export(output_file, format="wav")
    
    # print("ONE", output_file)


# def speech_file_to_array_fn(path, sampling_rate):
#     speech_array, _sampling_rate = torchaudio.load(path,format="mp3")
#     resampler = torchaudio.transforms.Resample(_sampling_rate)
#     speech = resampler(speech_array[1]).squeeze().numpy()
#     return speech


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

@app.post('/inference')
async def perform_inference(audio_file: UploadFile):
    
    # Convert the binary blob to a WAV file
    wav_output_file = "output.wav"  # Set the desired path for the output WAV file
    convert_blob_to_wav(await audio_file.read(), wav_output_file)

    # Already loaded the Hugging Face model

    # Process the audio using the model
    audio_input, _sampling_rate = torchaudio.load(wav_output_file)
    # print("PROCESS")
    
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(audio_input[0]).squeeze().numpy()
    
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        logits = model(input_values.input_values).logits

    # print("AFTER PROCESSOR")
    # Perform voice recognition
    transcription = processor.batch_decode(torch.argmax(logits, dim=-1))
    
    # result = my_inference_function(waveform)
    # print("TRANSCRIPTION", transcription)
    return {'result': transcription}
    # return {'result': 'hi'}

# Allow all origins (you can customize this to restrict to specific origins)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
   uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)