from django.db import models
from django.contrib.auth.models import AbstractUser
from django.db.models.fields import CharField


class Users(AbstractUser):
    name = models.CharField(max_length=225)
    email = models.CharField(max_length=225, unique=True)
    password = models.CharField(max_length=225)
    username = None

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []




# Create your models here.
