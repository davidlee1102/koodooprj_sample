from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.request import Request

from koodoovoice.ultils import response_message_process, logs_record
from koodoovoice.model_packages import voice_models


@api_view(["POST"])
def summary_test(request: Request):
    """Example
       {
            "conversation": "Hi ... this is ... thank you"
        }
        """
    data = request.data
    conversation_request = data.get("conversation", "")
    if not conversation_request:
        return Response(response_message_process.status_response('Error Input - Please Check Again'),
                        status=status.HTTP_400_BAD_REQUEST)
    else:
        result = voice_models.model_loader(conversation_request)
        logs_record.dataframe_records('summary', result, 'processed')
        return Response("OK", status=status.HTTP_200_OK)
