from rest_framework import serializers
from .models import Users


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = Users
        fields = ['id', 'name', 'email', 'password']
        extra_kwargs = {
            'password': {'write_only': True}
        }


def create(self, validated_data):
    password = validated_data.pop('password', None)
    isinstance = self.Meta.model(**validated_data)
    if password is not None:
        isinstance.set_password(password)
        isinstance.save()
        return isinstance
