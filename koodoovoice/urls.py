from django.urls import path
from . import views
from .views import health_check, voice_request, test_request
urlpatterns = [
    path("", health_check.health_check),
    path('disclaim_check/', voice_request.disclaimer_verification),
    path('voice_convert/', voice_request.covert_voice_to_wav),
    path('dialogue_convert/', voice_request.dialogue_convert),
    path('conversation_summary/', voice_request.conversation_summary),
    path('emotion_check/', voice_request.emotion_user_checking),
]