from django.db import models
from django.contrib.auth.models import User

class PredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)

    leaf_vibration = models.FloatField()
    pollen_scent_complexity = models.IntegerField()
    bioluminescence_intensity = models.FloatField()
    root_signal_strength = models.FloatField()
    growth_rate = models.FloatField()
    ambient_temperature = models.FloatField()
    soil_moisture = models.FloatField()
    sunlight_exposure = models.FloatField()
    symbiotic_fungus = models.IntegerField()

    result = models.CharField(max_length=100)
    best_model = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.result