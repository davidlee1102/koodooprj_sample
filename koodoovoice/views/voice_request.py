from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.request import Request

from koodoovoice.ultils import response_message_process
from koodoovoice.model_packages import voice_models

@api_view(["POST"])
def disclaimer_verification(request: Request):
    """Example
       {
           "coversation_process": "Your conversation need to check"
       }
       """
    data = request.data
    conversation_request = data.get("coversation_process", "")
    if not conversation_request:
        return Response(response_message_process.status_response('Error Input - Please Check Again'),
                        status=status.HTTP_400_BAD_REQUEST)
    else:
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
           "file_path": "Your wav file path"
       }
       """
    data = request.data
    file_path_request = data.get("file_path", "")
    if not file_path_request:
        return Response(response_message_process.status_response('Error Input - Please Check Again'),
                        status=status.HTTP_400_BAD_REQUEST)
    else:
        diarization = voice_models.diarization_convert(file_path_request, num_speaker=2)
        return Response(diarization, status=status.HTTP_200_OK)
