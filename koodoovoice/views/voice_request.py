from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.request import Request

from koodoovoice.ultils import response_message_process, logs_record
from koodoovoice.model_packages import voice_models, whisper_voice_model


@api_view(["POST"])
def disclaimer_verification(request: Request):
    """Example
       {
            "file_path": "koodoovoice/voice_data/Call-1-Example"
        }
        """
    data = request.data
    file_path_request = data.get("file_path", "")
    if not file_path_request:
        return Response(response_message_process.status_response('Error Input - Please Check Again'),
                        status=status.HTTP_400_BAD_REQUEST)
    else:
        diarization = voice_models.diarization_convert(file_path_request, num_speaker=2)
        transcriptions_by_speaker, dialogue_details = voice_models.extract_and_transcribe_segments(file_path_request,
                                                                                                   diarization)
        conversation_request = transcriptions_by_speaker['SPEAKER_00'].get("transcription", "")
        print(conversation_request)
        result = voice_models.check_disclaimer(str(conversation_request))
        return Response(result, status=status.HTTP_200_OK)


@api_view(["POST"])
def covert_voice_to_wav(request: Request):
    """Example
       {
           "file_path": "Your mp3 file path"
       }
       """
    data = request.data
    file_path_request = data.get("file_path", "")
    if not file_path_request:
        return Response(response_message_process.status_response('Error Input - Please Check Again'),
                        status=status.HTTP_400_BAD_REQUEST)
    else:
        result = voice_models.convert_to_wav(file_path_request)
        if result is True:
            return Response("Converted", status=status.HTTP_200_OK)
        else:
            return Response(response_message_process.status_response('Error During Processing - Please Check Again'),
                            status=status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
def dialogue_convert(request: Request):
    """Example
       {
            "file_path": "koodoovoice/voice_data/Call-1-Example"
        }
        """
    data = request.data
    file_path_request = data.get("file_path", "")
    if not file_path_request:
        return Response(response_message_process.status_response('Error Input - Please Check Again'),
                        status=status.HTTP_400_BAD_REQUEST)
    else:
        diarization = voice_models.diarization_convert(file_path_request, num_speaker=2)
        transcriptions_by_speaker, dialogue_details = voice_models.extract_and_transcribe_segments(file_path_request,
                                                                                                   diarization)
        logs_record.dataframe_records('transcription', transcriptions_by_speaker, 'processed')
        print("ok")
        return Response("OK", status=status.HTTP_200_OK)


@api_view(["POST"])
def conversation_summary(request: Request):
    """Example
       {
            "file_path": "koodoovoice/voice_data/Call-1-Example"
        }
        """
    data = request.data
    file_path_request = data.get("file_path", "")
    if not file_path_request:
        return Response(response_message_process.status_response('Error Input - Please Check Again'),
                        status=status.HTTP_400_BAD_REQUEST)
    else:
        diarization = voice_models.diarization_convert(file_path_request, num_speaker=2)
        _, dialogue_details = voice_models.extract_and_transcribe_segments(file_path_request, diarization)
        result = voice_models.model_loader(dialogue_details)
        logs_record.dataframe_records('summary', result, 'processed')
        return Response(result, status=status.HTTP_200_OK)


@api_view(["POST"])
def emotion_user_checking(request: Request):
    """Example
       {
            "file_path": "koodoovoice/voice_data/Call-1-Example"
        }
        """
    data = request.data
    file_path_request = data.get("file_path", "")
    if not file_path_request:
        return Response(response_message_process.status_response('Error Input - Please Check Again'),
                        status=status.HTTP_400_BAD_REQUEST)
    else:
        diarization = voice_models.diarization_convert(file_path_request, num_speaker=2)
        transcriptions_by_speaker, dialogue_details = voice_models.extract_and_transcribe_segments(file_path_request,
                                                                                                   diarization)
        file_path = voice_models.merge_and_play_speaker_segments(transcriptions_by_speaker, "SPEAKER_01")
        result, emotion = voice_models.voice_emotion_classify(file_path, transcriptions_by_speaker)
        data_str_add = emotion + result
        logs_record.dataframe_records('emotion_checking', data_str_add, 'processed')
        print(emotion, result)
        return Response(data_str_add, status=status.HTTP_200_OK)


@api_view(["POST"])
def whisper_emotion_user_checking(request: Request):
    """Example
       {
            "file_path": "koodoovoice/voice_data/Call-1-Example"
        }
        """
    data = request.data
    file_path_request = data.get("file_path", "")
    if not file_path_request:
        return Response(response_message_process.status_response('Error Input - Please Check Again'),
                        status=status.HTTP_400_BAD_REQUEST)
    else:
        diarization = whisper_voice_model.whisper_diarization_convert(file_path_request)
        transcriptions_by_speaker, dialogue_details = whisper_voice_model.speech_discriminate(diarization, 2,
                                                                                              file_path_request)
        file_path = voice_models.merge_and_play_speaker_segments(transcriptions_by_speaker, "SPEAKER_01")
        result, emotion = voice_models.voice_emotion_classify(file_path, transcriptions_by_speaker)
        data_str_add = emotion + result
        logs_record.dataframe_records('emotion_checking', data_str_add, 'processed')
        print(emotion, result)
        return Response(data_str_add, status=status.HTTP_200_OK)
