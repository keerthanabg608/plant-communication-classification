from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages

import numpy as np
import joblib
import os

from .models import PredictionHistory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))

label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
accuracies = joblib.load(os.path.join(BASE_DIR, "accuracies.pkl"))
best_model_name = joblib.load(os.path.join(BASE_DIR, "best_model_name.pkl"))


def index(request):
    return render(request, "index.html")


def register_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]
        password = request.POST["password"]

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            return redirect("register")

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )

        user.save()

        messages.success(request, "Registration Successful")
        return redirect("login")

    return render(request, "register.html")


def login_view(request):
    if request.method == "POST":

        username = request.POST["username"]
        password = request.POST["password"]

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is not None:
            login(request, user)
            return redirect("form")

        else:
            messages.error(request, "Invalid Username or Password")

    return render(request, "login.html")


def logout_view(request):
    logout(request)
    return redirect("login")


@login_required
def form_view(request):
    return render(request, "form.html")

@login_required
def predict(request):

    if request.method == "POST":

        leaf_vibration = float(request.POST["leaf_vibration"])
        pollen_scent_complexity = int(request.POST["pollen_scent_complexity"])
        bioluminescence_intensity = float(request.POST["bioluminescence_intensity"])
        root_signal_strength = float(request.POST["root_signal_strength"])
        growth_rate = float(request.POST["growth_rate"])
        ambient_temperature = float(request.POST["ambient_temperature"])
        soil_moisture = float(request.POST["soil_moisture"])
        sunlight_exposure = float(request.POST["sunlight_exposure"])
        symbiotic_fungus = int(request.POST["symbiotic_fungus"])

        data = [[
            leaf_vibration,
            pollen_scent_complexity,
            bioluminescence_intensity,
            root_signal_strength,
            growth_rate,
            ambient_temperature,
            soil_moisture,
            sunlight_exposure,
            symbiotic_fungus
        ]]

        prediction = model.predict(data)

        result = label_encoder.inverse_transform(prediction)[0]

        # OUTPUT MEANINGS

        meanings = {

            "Contentment":
            "The plant is healthy, stable, and growing in favorable environmental conditions.",

            "Stress Signal":
            "The plant may be under stress due to water deficiency, heat, or poor environmental conditions.",

            "Defense Alert":
            "The plant may be reacting to pests, diseases, or harmful surroundings.",

            "Growth Boost":
            "The plant is actively growing and responding positively to nutrients and sunlight.",

            "Nutrient Request":
            "The plant may require additional minerals, nutrients, or water for healthy growth.",

            "Pollination Attraction":
            "The plant is generating biological signals to attract pollinators.",

            "Symbiotic Communication":
            "The plant is interacting positively with fungi or surrounding organisms.",

            "Dormancy Signal":
            "The plant is entering a resting or low-growth stage.",

            "Environmental Warning":
            "The plant is reacting to extreme temperature, low moisture, or environmental imbalance.",

            "Energy Optimization":
            "The plant is adjusting energy usage to improve survival and growth efficiency."
        }

        meaning = meanings.get(
            result,
            "No meaning available for this prediction."
        )

        PredictionHistory.objects.create(
            user=request.user,
            leaf_vibration=leaf_vibration,
            pollen_scent_complexity=pollen_scent_complexity,
            bioluminescence_intensity=bioluminescence_intensity,
            root_signal_strength=root_signal_strength,
            growth_rate=growth_rate,
            ambient_temperature=ambient_temperature,
            soil_moisture=soil_moisture,
            sunlight_exposure=sunlight_exposure,
            symbiotic_fungus=symbiotic_fungus,
            result=result,
            best_model=best_model_name
        )

        return render(request, "result.html", {

            "result": result,
            "meaning": meaning,
            "accuracies": accuracies,
            "best_model": best_model_name
        })

    return redirect("form")
