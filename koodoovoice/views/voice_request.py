from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.request import Request
from django.http import HttpResponse
from django.shortcuts import render, redirect

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
        try:
            diarization = voice_models.diarization_convert(file_path_request, num_speaker=2)
        except Exception as E:
            print(E)
            return Response("Function Error, Please Check Your File", status=status.HTTP_400_BAD_REQUEST)
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
        try:
            diarization = voice_models.diarization_convert(file_path_request, num_speaker=2)
        except Exception as E:
            print(E)
            return Response("Function Error, Please Check Your File", status=status.HTTP_400_BAD_REQUEST)
        transcriptions_by_speaker, dialogue_details = voice_models.extract_and_transcribe_segments(file_path_request,
                                                                                                   diarization)
        logs_record.dataframe_records('transcription', transcriptions_by_speaker, 'processed')
        print("ok")
        return Response(str(transcriptions_by_speaker), status=status.HTTP_200_OK)


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
        try:
            diarization = voice_models.diarization_convert(file_path_request, num_speaker=2)
        except Exception as E:
            print(E)
            return Response("Function Error, Please Check Your File", status=status.HTTP_400_BAD_REQUEST)
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
        try:
            diarization = voice_models.diarization_convert(file_path_request, num_speaker=2)
        except Exception as E:
            print(E)
            return Response("Function Error, Please Check Your File", status=status.HTTP_400_BAD_REQUEST)

        transcriptions_by_speaker, dialogue_details = voice_models.extract_and_transcribe_segments(file_path_request,
                                                                                                   diarization)
        file_path = voice_models.merge_and_play_speaker_segments(transcriptions_by_speaker, "SPEAKER_01")
        result, emotion = voice_models.voice_emotion_classify(file_path, transcriptions_by_speaker)
        data_str_add = emotion + " " + result
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
        try:
            diarization = whisper_voice_model.whisper_diarization_convert(file_path_request)
        except Exception as E:
            print(E)
            return Response("Function Error, Please Check Your File", status=status.HTTP_400_BAD_REQUEST)
        transcriptions_by_speaker, dialogue_details = whisper_voice_model.speech_discriminate(diarization, 2,
                                                                                              file_path_request)
        file_path = voice_models.merge_and_play_speaker_segments(transcriptions_by_speaker, "SPEAKER_01")
        result, emotion = voice_models.voice_emotion_classify(file_path, transcriptions_by_speaker)
        data_str_add = emotion + result
        logs_record.dataframe_records('emotion_checking', data_str_add, 'processed')
        print(emotion, result)
        return Response(data_str_add, status=status.HTTP_200_OK)


def compare_models(request):
    default_value = 0
    if request.method == 'POST':
        button_value = request.POST.get('button_name')
        print(f'Button Value: {button_value}')
        if str(button_value) == "value_4":
            file_path_request = request.POST.get("file_path")
            diarization = voice_models.diarization_convert(file_path_request, num_speaker=2)
            _, dialogue_details = voice_models.extract_and_transcribe_segments(file_path_request, diarization)
            result_1 = voice_models.model_loader(dialogue_details)
            print(result_1)
            print("___")
            result_2 = result_1  # voice_models.model_loader(dialogue_details)
            print(result_2)
            logs_record.dataframe_records('summary', result_1, 'processed')
            logs_record.dataframe_records('summary', result_2, 'processed')
            request.session['model_one'] = result_1
            request.session['model_two'] = result_2
        elif str(button_value) == "value_1":
            result_1 = request.session.get('model_one')
            if result_1:
                print("Thank for submission")
                logs_record.dataframe_records('vote_summary', result_1, 'processed')

            context = {
                'model_one': 0,
                'model_two': 1,
            }
            return render(request, '/Users/davidlee/PycharmProjects/KoodooProject/templates/compare_models.html',
                          context)
        elif str(button_value) == "value_2":
            result_2 = request.session.get('model_two')
            if result_2:
                print("Thank for submission")
                logs_record.dataframe_records('vote_summary', result_2, 'processed')
            context = {
                'model_one': 0,
                'model_two': 1,
            }
            return render(request, '/Users/davidlee/PycharmProjects/KoodooProject/templates/compare_models.html',
                          context)
        elif str(button_value) == "value_3":
            user_summary = request.POST.get("user_summary")
            if user_summary:
                logs_record.dataframe_records('vote_summary', user_summary, 'processed')
            context = {
                'model_one': 0,
                'model_two': 1,
            }
            return render(request, '/Users/davidlee/PycharmProjects/KoodooProject/templates/compare_models.html',
                          context)

        if not result_1 or not result_2:
            context = {
                'model_one': 0,
                'model_two': 1,
            }
        else:
            context = {
                'model_one': result_1,
                'model_two': result_2,
            }
        return render(request, '/Users/davidlee/PycharmProjects/KoodooProject/templates/compare_models.html', context)


    else:
        context = {
            'model_one': 0,
            'model_two': 1,
        }
        return render(request, '/Users/davidlee/PycharmProjects/KoodooProject/templates/compare_models.html', context)
