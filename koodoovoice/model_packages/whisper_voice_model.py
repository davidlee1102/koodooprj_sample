import torch
import wave
import whisper
import datetime
import subprocess
import contextlib
import numpy as np
import pyannote.audio

from pydub import AudioSegment
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering, KMeans
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from koodoovoice.model_packages import constant_key

embedding_model = PretrainedSpeakerEmbedding(constant_key.WHISPER_EMBEDDING_PATH)
model_name = constant_key.WHISPER_MODEL_NAME


def whisper_diarization_convert(file_path):
    model = whisper.load_model(model_name)
    diarization = model.transcribe(file_path)
    diarization = diarization.get("segments", "")
    return diarization


def get_duration(file_path):
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration


def segment_embedding(segment, file_path, duration):
    audio = Audio()
    audio_segment = AudioSegment.from_file(file_path)
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(file_path, clip)
    audio_record = audio_segment[start:end]
    mono_waveform = torch.mean(waveform, dim=0, keepdim=True)
    return audio_record, embedding_model(mono_waveform)


def speech_discriminate(segments, num_speakers, file_path):
    embeddings = np.zeros(shape=(len(segments), 512))
    audio_list = []
    dialogue_details = []
    transcriptions_by_speaker = {}
    duration = get_duration(file_path)
    for i, segment in enumerate(segments):
        audio, embeddings[i] = segment_embedding(segment, file_path, duration)
        audio_list.append(audio)
    embeddings = np.nan_to_num(embeddings)
    clustering = KMeans(num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        speaker_id = segments[i]["speaker"] = "SPEAKER_" + '0' + str(labels[i])
        # voice_start = segments[i].get("start", "")
        # voice_end = segments[i].get("end", "")
        voice_text = segments[i].get("text", "")
        if speaker_id not in transcriptions_by_speaker:
            transcriptions_by_speaker[speaker_id] = {"transcription": "", "segments": []}
        transcriptions_by_speaker[speaker_id]["transcription"] += " " + voice_text
        transcriptions_by_speaker[speaker_id]["segments"].append(audio_list[i])

        dialogue_detail = {"speaker": speaker_id, "transcription": voice_text}
        dialogue_details.append(dialogue_detail)
        # print(f"speaker: {speaker_id} - start: {voice_start} - end: {voice_end} - text: {voice_text}")
    return transcriptions_by_speaker, dialogue_details
