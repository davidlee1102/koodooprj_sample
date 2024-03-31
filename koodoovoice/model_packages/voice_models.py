import os
import spacy
import torch
import shutil
import time
import speech_recognition as sr
import transformers
import numpy as np

from pydub import AudioSegment
from pyannote.audio import Pipeline

from koodoovoice.model_packages import constant_key
from speechbrain.inference.interfaces import foreign_class
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(constant_key.TOKENIZER_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(constant_key.MODEL_PATH)
torch.mps.set_per_process_memory_fraction(0.0)


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
    print("oke, process")
    diarization = pipeline(file_path, num_speakers=2)
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

        dialogue_detail = {"speaker": speaker, "transcription": transcription}
        # "start": speech_turn.start, "end": speech_turn.end
        dialogue_details.append(dialogue_detail)

        # print(f"{speech_turn.start:4.5f} - {speech_turn.end:4.5f} {speaker}: {transcription}")
    return transcriptions_by_speaker, dialogue_details


def merge_and_play_speaker_segments(transcriptions_by_speaker, speaker_id, output_dir="koodoovoice/voice_data_merged"):
    """
    Merges all audio segments for a given speaker ID and plays the merged audio in Google Colab.

    Args:
        transcriptions_by_speaker (dict): Dictionary containing aggregated transcriptions and audio segments for each speaker.
        speaker_id (str): The ID of the speaker whose segments are to be merged and played.
        output_dir (str): Directory where the merged audio segment will be saved.
    """
    if speaker_id not in transcriptions_by_speaker:
        print(f"No segments found for {speaker_id}")
        return

    # Concatenate all segments for the specified speaker
    merged_segment = AudioSegment.silent(duration=0)  # Start with a silent segment to concatenate to
    for segment in transcriptions_by_speaker[speaker_id]['segments']:
        merged_segment += segment

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the merged segment to a file
    merged_filename = os.path.join(output_dir, f"{speaker_id}_merged.wav")
    merged_segment.export(merged_filename, format="wav")
    return merged_filename


def model_loader(conversation_summary):
    collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)
    args = transformers.Seq2SeqTrainingArguments(
        'conversation-summ',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        eval_accumulation_steps=1,
        fp16=False,
        use_mps_device=True,
    )

    trainer = transformers.Seq2SeqTrainer(
        model,
        args,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    try:
        model_inputs = tokenizer(text_processing_summary(conversation_summary), max_length=512, padding='max_length',
                                 truncation=True)
    except:
        model_inputs = tokenizer(conversation_summary, max_length=512, padding='max_length',
                                 truncation=True)
    raw_pred, _, _ = trainer.predict([model_inputs])
    result = tokenizer.decode(raw_pred[0], skip_special_tokens=True)
    return str(result)


def text_processing_summary(dialogue_details):
    conversation = ""
    for i in range(len(dialogue_details)):
        speaker_id = dialogue_details[i].get("speaker")
        transcription = dialogue_details[i].get("transcription")
        conversation = conversation + speaker_id + ":" + transcription
    return conversation


classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                           pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")


def voice_emotion_classify(file_path, transcriptions_by_speaker):
    out_prob, score, index, text_lab = classifier.classify_file(file_path)
    result = ""
    if text_lab == 'hap':
        return result, 'positive'
    elif text_lab == 'neu':
        return result, 'neutral'
    else:
        user_conversation = '.'.join(transcriptions_by_speaker['SPEAKER_01'].get("transcription", "").split(".")[:2])
        result = model_loader(user_conversation)
        return result, 'negative'
