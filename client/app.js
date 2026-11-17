// Predicts in the browser from model/model.json, the same weights the Flask API uses
// (src/homevalue/model.py). The checks below mirror that module's, so the page and the
// API refuse the same inputs. Text from the model file is set with textContent, never
// parsed as HTML.
"use strict";

const MODEL_URL = "../model/model.json";
const MIN_SQFT_PER_BHK = 300;
const MAX_SQFT = 30000;
const CHOICES = 5;

const form = document.getElementById("form");
const result = document.getElementById("result");
let model = null;

function show(message, isError) {
    result.textContent = message;
    result.classList.toggle("error", isError);
}

function addChoices(containerId, name) {
    const container = document.getElementById(containerId);
    for (let i = 1; i <= CHOICES; i += 1) {
        const input = Object.assign(document.createElement("input"), {
            type: "radio", name, id: `${name}-${i}`, value: String(i), checked: i === 2,
        });
        const label = Object.assign(document.createElement("label"), {
            htmlFor: input.id, textContent: String(i),
        });
        container.append(input, label);
    }
}

function fillLocations(names) {
    const select = document.getElementById("location");
    select.replaceChildren(new Option("Choose a location", ""));
    names.forEach((name) => select.add(new Option(name, name)));
}

function chosen(name) {
    const checked = form.querySelector(`input[name="${name}"]:checked`);
    return checked ? Number(checked.value) : NaN;
}

function refusal(sqft, bhk, bath, location) {
    if (!Number.isFinite(sqft)) return "Enter the area as a number.";
    if (sqft > MAX_SQFT) return `The area must be at most ${MAX_SQFT.toLocaleString("en-IN")} sq ft.`;
    if (sqft < MIN_SQFT_PER_BHK * bhk) {
        return `The area must be at least ${MIN_SQFT_PER_BHK} sq ft per bedroom `
            + `(${MIN_SQFT_PER_BHK * bhk} for ${bhk} BHK).`;
    }
    if (bath > bhk + 1) return "Bathrooms can be at most one more than bedrooms.";
    if (!Object.hasOwn(model.locations, location)) return "Choose a location from the list.";
    return null;
}

function estimate(sqft, bhk, bath, location) {
    const w = model.coefficients;
    return model.intercept + w.total_sqft * sqft + w.bath * bath + w.bhk * bhk
        + model.locations[location];
}

form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!model) return show("The model has not loaded yet.", true);
    const sqft = Number(document.getElementById("sqft").value);
    const bhk = chosen("bhk");
    const bath = chosen("bath");
    const location = document.getElementById("location").value;
    const why = refusal(sqft, bhk, bath, location);
    if (why) return show(why, true);
    const price = estimate(sqft, bhk, bath, location);
    if (!(price > 0)) return show("These inputs are outside what the model was trained on.", true);
    const error = model.scores.mae_held_out_lakh;
    return show(`About ${price.toFixed(2)} lakh rupees, typically within ${error.toFixed(0)} lakh`, false);
});

addChoices("bhk", "bhk");
addChoices("bath", "bath");
fetch(MODEL_URL)
    .then((response) => {
        if (!response.ok) throw new Error(String(response.status));
        return response.json();
    })
    .then((loaded) => {
        model = loaded;
        fillLocations(Object.keys(loaded.locations).filter((n) => n !== "other").sort());
    })
    .catch(() => show("The model could not be loaded. Try reloading the page.", true));
