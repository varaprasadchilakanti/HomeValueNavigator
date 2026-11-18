// SPDX-License-Identifier: MIT
// Estimates in the browser from the model's weights (model/model.js, written with
// model/model.json by src/homevalue/train.py). The checks follow src/homevalue/model.py
// and read the same bounds; the page offers 1 to 5 bedrooms and opens on the location
// with the most listings. Text is set with
// textContent only.
"use strict";

const CHOICES = 5;
const MAX_OPTIONS = 8;
const ANNOUNCE_DELAY_MS = 500;
const RUPEE = "\u20B9";
const DASH = "\u2013";
const OTHER = "other";
const LAKH = 100000;
const RATE_STEP = 50; // rupees; the rate is an estimate, not a quote
const EM_DASH = "\u2014";

const byId = (id) => document.getElementById(id);
const key = (name) => name.trim().split(/\s+/).join(" ").toLowerCase();
const isNumber = (v) => typeof v === "number" && Number.isFinite(v);

// ---------------------------------------------------------------- model

/** The weights and bounds from model.js, or null when the file is missing or malformed. */
function loadModel(raw) {
    const w = raw?.coefficients;
    const b = raw?.bounds;
    const within = raw?.scores?.held_out?.median_abs_pct_error;
    const tested = raw?.scores?.held_out?.rows;
    const numbers = [raw?.intercept, w?.log_total_sqft, w?.bath, w?.bhk, within, tested,
        b?.min_sqft_per_bhk, b?.max_sqft, b?.max_extra_bath];
    if (!numbers.every(isNumber) || typeof raw.locations !== "object" || !raw.locations) return null;
    const offsets = new Map(Object.entries(raw.locations).filter(([, v]) => isNumber(v)));
    const names = [...offsets.keys()].filter((n) => n !== OTHER).sort((a, c) => a.localeCompare(c));
    if (!names.includes(raw.default_location)) return null;
    return {
        estimate: (sqft, bhk, bath, place) => Math.exp(raw.intercept + w.log_total_sqft * Math.log(sqft)
            + w.bath * bath + w.bhk * bhk + offsets.get(place)),
        names, within, tested, defaultLocation: raw.default_location,
        minPerBhk: b.min_sqft_per_bhk, maxSqft: b.max_sqft, extraBath: b.max_extra_bath,
    };
}

// ---------------------------------------------------------------- theme

function applyTheme(choice) {
    const root = document.documentElement;
    root.classList.add("no-transitions");
    if (choice === "system") delete root.dataset.theme;
    else root.dataset.theme = choice;
    try {
        if (choice === "system") localStorage.removeItem("theme");
        else localStorage.setItem("theme", choice);
    } catch { /* not saved; applies to this visit */ }
    void getComputedStyle(root).opacity; // paint the new colours before transitions return
    root.classList.remove("no-transitions");
}

function setUpTheme() {
    byId(`theme-${document.documentElement.dataset.theme ?? "system"}`).checked = true;
    document.querySelectorAll('input[name="theme"]').forEach((input) => {
        input.addEventListener("change", () => applyTheme(input.value));
    });
}

// ---------------------------------------------------------------- formatting

/** 69.7 -> "Rs 70 lakh"; from 100 lakh, crore to one place. Finer would claim precision
 *  the model does not have. */
function money(lakh) {
    if (lakh >= 99.5) return `${RUPEE}${(lakh / 100).toFixed(1)} crore`;
    return `${RUPEE}${Math.round(lakh)} lakh`;
}

/** "Rs 56 - 83 lakh", or both units in full when they differ. */
function span(low, high) {
    const [a, b] = [money(low), money(high)];
    const [aNumber, aUnit] = a.split(" ");
    const [bNumber, bUnit] = b.split(" ");
    if (aUnit === bUnit) return `${aNumber} ${DASH} ${bNumber.slice(RUPEE.length)} ${bUnit}`;
    return `${a} ${DASH} ${b}`;
}

/** Digits, optionally with commas, spaces or one decimal point; anything else is NaN. */
function parseArea(text) {
    const cleaned = text.replace(/[,\s]/g, "");
    return /^\d+(\.\d+)?$/.test(cleaned) ? Number(cleaned) : NaN;
}

// ---------------------------------------------------------------- the page

function main() {
    const model = loadModel(window.HOMEVALUE_MODEL);
    if (!model) {
        byId("range").textContent = "The model did not load. Try reloading the page.";
        document.querySelectorAll("#form input").forEach((input) => { input.disabled = true; });
        return;
    }
    const byKey = new Map(model.names.map((n) => [key(n), n]));
    const touched = { sqft: false, location: false };
    const location = byId("location");
    const list = byId("location-list");
    let announceTimer = 0;
    let active = -1;
    let shown = [];

    function addSegments(name, start) {
        for (let i = 1; i <= CHOICES; i += 1) {
            const input = Object.assign(document.createElement("input"), {
                type: "radio", name, id: `${name}-${i}`, value: String(i), checked: i === start,
                className: "sr-only",
            });
            const label = Object.assign(document.createElement("label"), {
                htmlFor: input.id, textContent: String(i),
            });
            byId(name).append(input, label);
        }
    }

    function chosen(name) {
        return Number(document.querySelector(`input[name="${name}"]:checked`)?.value);
    }

    // Bathroom counts the training data did not cover are not offered.
    function limitBaths() {
        const most = chosen("bhk") + model.extraBath;
        document.querySelectorAll('input[name="bath"]').forEach((input) => {
            input.disabled = Number(input.value) > most;
            if (input.disabled && input.checked) byId(`bath-${most}`).checked = true;
        });
    }

    function setError(id, message) {
        const box = byId(`${id}-error`);
        box.hidden = message === null;
        const prefix = Object.assign(document.createElement("span"), { className: "sr-only", textContent: "Error: " });
        box.replaceChildren(...(message === null ? [] : [prefix, message]));
        byId(id).setAttribute("aria-invalid", String(message !== null));
    }

    function areaProblem(sqft, bhk) {
        if (!Number.isFinite(sqft)) return "Enter the area in square feet, for example 1200.";
        if (sqft > model.maxSqft) return `Enter an area up to ${model.maxSqft.toLocaleString("en-IN")} sq ft.`;
        const least = model.minPerBhk * bhk;
        if (sqft < least) return `A ${bhk} BHK home needs at least ${least.toLocaleString("en-IN")} sq ft.`;
        return null;
    }

    function show(price, range, spoken, perSqft = "") {
        byId("price").textContent = price;
        byId("price").classList.toggle("waiting", price === EM_DASH);
        byId("range").textContent = range;
        byId("per-sqft").textContent = perSqft;
        clearTimeout(announceTimer);
        announceTimer = setTimeout(() => { byId("announce").textContent = spoken; }, ANNOUNCE_DELAY_MS);
    }

    function update() {
        const sqft = parseArea(byId("sqft").value);
        const bhk = chosen("bhk");
        const typed = location.value;
        const place = byKey.get(key(typed));
        const area = areaProblem(sqft, bhk);
        setError("sqft", touched.sqft ? area : null);
        setError("location", touched.location && typed.trim() && !place ? "Choose a location from the list." : null);
        if (area || !place) return show(EM_DASH, "Enter the details to see an estimate.", "");
        const price = model.estimate(sqft, bhk, chosen("bath"), place);
        const low = price * (1 - model.within);
        const high = price * (1 + model.within);
        const rate = (Math.round((price * LAKH) / sqft / RATE_STEP) * RATE_STEP).toLocaleString("en-IN");
        return show(money(price), `Typical range ${span(low, high)}`,
            `Estimated asking price ${money(price)}; typical range ${money(low)} to ${money(high)}.`,
            `About ${RUPEE}${rate} per sq ft`);
    }

    // ---- location search: the WAI-ARIA combobox pattern, so the list follows the theme
    function setOpen(open) {
        list.hidden = !open;
        location.setAttribute("aria-expanded", String(open));
        if (!open) { active = -1; location.removeAttribute("aria-activedescendant"); }
    }

    function matches(query) {
        const q = key(query);
        if (!q) return [];
        const starts = model.names.filter((n) => key(n).startsWith(q));
        const within = model.names.filter((n) => !key(n).startsWith(q) && key(n).includes(q));
        return [...starts, ...within].slice(0, MAX_OPTIONS);
    }

    function option(name, index, q) {
        const item = Object.assign(document.createElement("li"), { id: `location-option-${index}` });
        item.setAttribute("role", "option");
        item.setAttribute("aria-selected", String(index === active));
        const at = key(name).indexOf(q);
        const mark = Object.assign(document.createElement("mark"), { textContent: name.slice(at, at + q.length) });
        item.append(name.slice(0, at), mark, name.slice(at + q.length));
        item.addEventListener("mousedown", (e) => e.preventDefault()); // keep focus in the input
        item.addEventListener("click", () => pick(name));
        return item;
    }

    function render() {
        shown = matches(location.value);
        if (!location.value.trim()) return setOpen(false);
        const q = key(location.value);
        list.replaceChildren(...(shown.length > 0
            ? shown.map((name, i) => option(name, i, q))
            : [Object.assign(document.createElement("li"), { className: "empty", textContent: "No matching location" })]));
        if (active >= 0) location.setAttribute("aria-activedescendant", `location-option-${active}`);
        return setOpen(true);
    }

    function pick(name) {
        location.value = name;
        touched.location = true;
        setOpen(false);
        update();
    }

    location.addEventListener("input", () => { active = -1; render(); update(); });
    location.addEventListener("keydown", (e) => {
        if ((e.key === "ArrowDown" || e.key === "ArrowUp") && shown.length > 0) {
            e.preventDefault();
            active = (active + (e.key === "ArrowDown" ? 1 : -1) + shown.length) % shown.length;
            render();
            byId(`location-option-${active}`)?.scrollIntoView({ block: "nearest" });
        } else if (e.key === "Enter" && !list.hidden && shown.length > 0) {
            e.preventDefault();
            pick(shown[Math.max(active, 0)]);
        } else if (e.key === "Escape" && !list.hidden) {
            e.preventDefault();
            setOpen(false);
        }
    });
    location.addEventListener("blur", () => {
        setOpen(false);
        const exact = byKey.get(key(location.value));
        if (exact) location.value = exact;
        touched.location = location.value.trim() !== "";
        update();
    });

    // ---- the rest of the form
    addSegments("bhk", 2);
    addSegments("bath", 2);
    limitBaths();
    byId("sqft").addEventListener("blur", () => { touched.sqft = true; update(); });
    byId("form").addEventListener("input", (e) => {
        if (e.target === location) return;
        if (e.target.name === "bhk") limitBaths();
        update();
    });
    byId("form").addEventListener("submit", (e) => e.preventDefault());
    location.placeholder = `Search ${model.names.length} locations`;
    location.value = model.defaultLocation;
    byId("accuracy").textContent = `Tested on ${model.tested.toLocaleString("en-IN")} listings it had `
        + `not seen, half of its estimates were within ${Math.round(model.within * 100)}% of the `
        + "asking price. The typical range is that band; the other half fall outside it.";
    update();
}

setUpTheme();
main();
