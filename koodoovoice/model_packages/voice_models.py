import os
import spacy
import torch
import shutil
import speech_recognition as sr

from pydub import AudioSegment
from pyannote.audio import Pipeline

from koodoovoice.model_packages import constant_key


def check_disclaimer(representative_transcript):
    """

    :param representative_transcript:
    :return:
    """
    nlp = spacy.load("en_core_web_sm")
    disclaimer_transcript = constant_key.VOICE_DISCLAIMER
    conversa_1 = nlp(disclaimer_transcript)
    conversa_2 = nlp(representative_transcript)
    similarity_score = conversa_1.similarity(conversa_2)
    print(similarity_score)
    if similarity_score > 0.8:
        result = "Disclaimer or a variation is likely present."
    else:
        result = "Disclaimer not found."
    print(result)
    return result


def convert_to_wav(file_path):
    # assert os.path.exists(file_path) is False, "File Path Does NOT Exist - Please Check"
    file_name = os.path.basename(file_path).split('/')[-1]
    file_name = file_name.replace("mp3", "")
    sound_covert = AudioSegment.from_mp3(file_path)
    data_path = constant_key.VOICE_PROCESS_PATH + "/" + file_name
    sound_covert.export(data_path, format="wav")

    return True


def transcribe_audio_from_segment(segment):
    """
    Transcribes a given audio segment using Google's Speech Recognition API.

    Args:
        segment (AudioSegment): An AudioSegment instance to transcribe.

    Returns:
        str: The transcribed text.
    """
    recognizer = sr.Recognizer()
    with segment.export(format="wav") as segment_file:
        with sr.AudioFile(segment_file) as source:
            audio_data = recognizer.record(source)
            try:
                return recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                return ""
            except sr.RequestError:
                return ""


def diarization_convert(file_path, num_speaker=2):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=constant_key.HF_KEY)
    diarization = pipeline("/content/Call-1-Example", num_speakers=2)
    return diarization


def extract_and_transcribe_segments(audio_file, diarization, output_dir="segments"):
    """
    Extracts segments from an audio file based on diarization results and transcribes them.
    Aggregates transcriptions by speaker.

    Args:
        audio_file (str): Path to the original audio file.
        diarization: Diarization result from Pyannote.audio.
        output_dir (str, optional): Directory to save the audio segments. Defaults to "segments".

    Returns:
        dict: A dictionary containing aggregated transcriptions and details for each speaker.
    """
    audio = AudioSegment.from_file(audio_file)
    dialogue_details = []
    transcriptions_by_speaker = {}

    for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
        start_ms, end_ms = int(speech_turn.start * 1000), int(speech_turn.end * 1000)
        segment = audio[start_ms:end_ms]

        transcription = transcribe_audio_from_segment(segment)
        transcription = transcription + "."

        if speaker not in transcriptions_by_speaker:
            transcriptions_by_speaker[speaker] = {"transcription": "", "segments": []}
        transcriptions_by_speaker[speaker]["transcription"] += " " + transcription
        transcriptions_by_speaker[speaker]["segments"].append(segment)

        dialogue_detail = {"start": speech_turn.start, "end": speech_turn.end, "speaker": speaker,
                           "transcription": transcription}
        dialogue_details.append(dialogue_detail)

        # print(f"{speech_turn.start:4.5f} - {speech_turn.end:4.5f} {speaker}: {transcription}")

    return transcriptions_by_speaker, dialogue_details
