// SPDX-License-Identifier: MIT
// Estimates in the browser. model/scopes.js lists each trained city with its location
// names; one search field covers them all, grouped by city, and choosing a location loads
// that city's weights (model/<id>/model.js, written with model.json by
// src/homevalue/train.py). The checks follow src/homevalue/model.py and read the same
// bounds. Nothing is chosen for the visitor; the answers are kept in the address, so a
// link or a reload shows the same estimate. Text is set with textContent only.
"use strict";

const CHOICES = 5;
const ANNOUNCE_DELAY_MS = 500;
const RUPEE = "\u20B9";
const DASH = "\u2013";
const LAKH = 100000;
const RATE_STEP = 50; // rupees; the rate is an estimate, not a quote
const EM_DASH = "\u2014";
const SCOPE_ID = /^[a-z0-9-]+$/; // a folder name; it becomes part of a script address

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
        b?.min_sqft_per_bhk, b?.max_extra_bath];
    if (!numbers.every(isNumber) || typeof raw.locations !== "object" || !raw.locations) return null;
    if (typeof b.max_sqft_by_bhk !== "object" || !b.max_sqft_by_bhk) return null;
    const largest = new Map(Object.entries(b.max_sqft_by_bhk)
        .filter(([, v]) => isNumber(v)).map(([k, v]) => [Number(k), v]));
    const offsets = new Map(Object.entries(raw.locations).filter(([, v]) => isNumber(v)));
    if (largest.size === 0) return null;
    // The page formats rupees in lakh and crore, and areas in sq ft; another scope's units
    // need their own formatting before this page can show it.
    const scope = raw.scope;
    if (scope?.currency !== "INR" || scope?.price_unit !== "lakh" || scope?.area_unit !== "sq ft"
        || typeof scope?.id !== "string") return null;
    return {
        estimate: (sqft, bhk, bath, place) => Math.exp(raw.intercept + w.log_total_sqft * Math.log(sqft)
            + w.bath * bath + w.bhk * bhk + offsets.get(place)),
        within, tested, scopeId: scope.id,
        minPerBhk: b.min_sqft_per_bhk, largest, extraBath: b.max_extra_bath,
    };
}

/** The well-formed cities in scopes.js, each with its location names; the rest are dropped. */
function readScopes(raw) {
    return (Array.isArray(raw) ? raw : []).filter((s) => typeof s?.id === "string" && SCOPE_ID.test(s.id)
        && typeof s.name === "string" && Array.isArray(s.locations)
        && s.locations.every((n) => typeof n === "string"));
}

/** One entry per location; labelled with its city once there is more than one city. */
function placesOf(scopes) {
    const many = scopes.length > 1;
    return scopes.flatMap((s) => s.locations.map((name) => ({
        scope: s.id, city: s.name, name, label: many ? `${name}, ${s.name}` : name,
    })));
}

/** The places for what is typed, cities in their order: every place when nothing is typed
 *  or a place is named exactly; otherwise those whose name or city starts with the text,
 *  then those whose label contains it. */
function matchPlaces(places, query, exact) {
    const q = key(query);
    if (!q || exact(q)) return places;
    const starts = (p) => key(p.name).startsWith(q) || key(p.city).startsWith(q);
    const ranked = [...places.filter(starts), ...places.filter((p) => !starts(p) && key(p.label).includes(q))];
    const cities = [...new Set(places.map((p) => p.scope))];
    return cities.flatMap((id) => ranked.filter((p) => p.scope === id));
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
    void getComputedStyle(root).opacity; // paint the new colors before transitions return
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

/** ASCII digits and one optional decimal part, once spaces at the ends and thousands commas
 *  are removed; anything else is NaN. The same rule as _number in src/homevalue/model.py. */
function parseArea(text) {
    const cleaned = text.trim().replaceAll(",", "");
    return /^[0-9]+(\.[0-9]+)?$/.test(cleaned) ? Number(cleaned) : NaN;
}

// ---------------------------------------------------------------- the page

function failed() {
    byId("range").textContent = "The model did not load. Try reloading the page.";
    document.querySelectorAll("#form input").forEach((input) => { input.disabled = true; });
}

/** A city's weights, loaded once from model/<id>/model.js; null when missing or malformed. */
function loadScope(id) {
    return new Promise((resolve) => {
        const script = Object.assign(document.createElement("script"), { src: `../model/${id}/model.js` });
        script.addEventListener("load", () => {
            const model = loadModel(window.HOMEVALUE_MODEL);
            resolve(model && model.scopeId === id ? model : null);
        });
        script.addEventListener("error", () => resolve(null));
        document.body.append(script);
    });
}

function start() {
    setUpTheme();
    const scopes = readScopes(window.HOMEVALUE_SCOPES);
    if (scopes.length === 0) return failed();
    return main(scopes);
}

function main(scopes) {
    const places = placesOf(scopes);
    const byKey = new Map(places.map((p) => [key(p.label), p]));
    for (const p of places) {
        if (places.filter((o) => key(o.name) === key(p.name)).length === 1) byKey.set(key(p.name), p);
    }
    const models = new Map(); // city id -> its model, or a Promise while it loads
    const touched = { sqft: false, location: false };
    const location = byId("location");
    const list = byId("location-list");
    const toggle = byId("location-toggle");
    let announceTimer = 0;
    let active = -1;
    let shown = [];
    let chosen = null; // the place picked last

    const ready = () => {
        const model = chosen ? models.get(chosen.scope) : null;
        return model && !(model instanceof Promise) ? model : null;
    };

    function addSegments(name, counts) {
        for (const i of counts) {
            const input = Object.assign(document.createElement("input"), {
                type: "radio", name, id: `${name}-${i}`, value: String(i), className: "sr-only",
            });
            const label = Object.assign(document.createElement("label"), { htmlFor: input.id, textContent: String(i) });
            // A radio cannot be cleared by itself: a second click on the chosen count clears it.
            input.addEventListener("click", () => {
                if (input.dataset.was !== "true") return;
                input.checked = false;
                if (name === "bhk") limitChoices();
                update();
            });
            byId(name).append(input, label);
        }
    }

    const picked = (name) => Number(document.querySelector(`input[name="${name}"]:checked`)?.value ?? NaN);

    // Counts the chosen city's data did not cover are not offered; nothing is chosen for the visitor.
    function limitChoices() {
        const model = ready();
        const bhk = picked("bhk");
        document.querySelectorAll('input[name="bhk"]').forEach((input) => {
            input.disabled = model !== null && !model.largest.has(Number(input.value));
            if (input.disabled) input.checked = false;
        });
        const most = model && Number.isFinite(bhk) ? bhk + model.extraBath : Infinity;
        document.querySelectorAll('input[name="bath"]').forEach((input) => {
            input.disabled = Number(input.value) > most;
            if (input.disabled) input.checked = false;
        });
    }

    function setError(id, message) {
        const box = byId(`${id}-error`);
        box.hidden = message === null;
        const prefix = Object.assign(document.createElement("span"), { className: "sr-only", textContent: "Error: " });
        box.replaceChildren(...(message === null ? [] : [prefix, message]));
        byId(id).setAttribute("aria-invalid", String(message !== null));
    }

    function areaProblem(model, sqft, bhk) {
        if (!Number.isFinite(sqft)) return "Enter the area in square feet, for example 1200.";
        if (!model || !Number.isFinite(bhk)) return null;
        const most = model.largest.get(bhk);
        if (sqft > most) {
            return `The largest ${bhk} BHK listing in ${chosen.city} is ${most.toLocaleString("en-IN")} sq ft; enter an area up to that.`;
        }
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

    // What is still needed, in the order the form asks for it.
    function missing(here, sqft, bhk, bath) {
        const need = [[!here, "a location"], [!Number.isFinite(bhk), "bedrooms"],
            [!Number.isFinite(bath), "bathrooms"], [!Number.isFinite(sqft), "the area"]];
        const names = need.filter(([gap]) => gap).map(([, name]) => name);
        if (names.length < 2) return names.join("");
        return `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;
    }

    function remember(here, sqft, bhk, bath) {
        const query = new URLSearchParams();
        if (here) { query.set("city", here.scope); query.set("location", here.name); }
        if (Number.isFinite(bhk)) query.set("bhk", String(bhk));
        if (Number.isFinite(bath)) query.set("bath", String(bath));
        if (Number.isFinite(sqft)) query.set("sqft", String(sqft));
        const text = query.toString();
        try {
            history.replaceState(null, "", text ? `?${text}` : window.location.pathname);
        } catch { /* a page opened from disk may refuse; the estimate still shows */ }
    }

    function update() {
        // The place counts only while the field still names it; a half-typed edit does not.
        const here = byKey.get(key(location.value)) === chosen ? chosen : null;
        const model = here ? ready() : null;
        const sqft = parseArea(byId("sqft").value);
        const [bhk, bath] = [picked("bhk"), picked("bath")];
        document.querySelectorAll(".segments input").forEach((input) => { input.dataset.was = String(input.checked); });
        const typed = location.value;
        const area = areaProblem(model, sqft, bhk);
        const sqftBox = byId("sqft");
        setError("sqft", touched.sqft && (sqftBox.value.trim() || sqftBox.validity.badInput) ? area : null);
        setError("location", touched.location && typed.trim() && !byKey.get(key(typed))
            ? "Choose a location from the list." : null);
        remember(here, sqft, bhk, bath);
        byId("accuracy").textContent = model ? `Tested on ${model.tested.toLocaleString("en-IN")} ${chosen.city} `
            + `listings it had not seen, half of its estimates were within ${Math.round(model.within * 100)}% `
            + "of the asking price. The typical range is that band; the other half fall outside it." : "";
        const gaps = missing(here, sqft, bhk, bath);
        if (gaps) return show(EM_DASH, `Choose ${gaps} to see an estimate.`, "");
        if (!model) return show(EM_DASH, `Loading ${chosen.city}${EM_DASH}`, "");
        if (area) return show(EM_DASH, "Correct the area to see an estimate.", "");
        const price = model.estimate(sqft, bhk, bath, chosen.name);
        const low = price * (1 - model.within);
        const high = price * (1 + model.within);
        const rate = (Math.round((price * LAKH) / sqft / RATE_STEP) * RATE_STEP).toLocaleString("en-IN");
        return show(money(price), `Typical range ${span(low, high)}`,
            `Estimated asking price ${money(price)}; typical range ${money(low)} to ${money(high)}.`,
            `About ${RUPEE}${rate} per sq ft`);
    }

    // Loads the place's city when it is not loaded yet, then refreshes the form.
    function choose(place) {
        chosen = place;
        location.value = place.label;
        if (!models.has(place.scope)) {
            models.set(place.scope, loadScope(place.scope).then((model) => {
                models.set(place.scope, model);
                if (!model) setError("location", `${place.city}'s model did not load. Try reloading the page.`);
                limitChoices();
                update();
            }));
        }
        limitChoices();
        update();
    }

    // ---- location: an editable WAI-ARIA combobox with list autocomplete and an open button
    // (APG combobox pattern), its options grouped by city. Typing filters by place or city;
    // a click, the button or Down shows the list; only a listed place is accepted.
    function setOpen(open) {
        list.hidden = !open;
        location.setAttribute("aria-expanded", String(open));
        toggle.setAttribute("aria-expanded", String(open));
        if (!open) { active = -1; location.removeAttribute("aria-activedescendant"); }
    }

    function option(place, index, q) {
        const item = Object.assign(document.createElement("li"), { id: `location-option-${index}` });
        item.setAttribute("role", "option");
        item.setAttribute("aria-selected", "false");
        const at = q ? key(place.name).indexOf(q) : -1;
        if (at < 0) item.append(place.name);
        else {
            const mark = Object.assign(document.createElement("mark"), { textContent: place.name.slice(at, at + q.length) });
            item.append(place.name.slice(0, at), mark, place.name.slice(at + q.length));
        }
        item.addEventListener("mousedown", (e) => e.preventDefault()); // keep focus in the input
        item.addEventListener("click", () => pick(place));
        return item;
    }

    // One group per city, named by a heading the listbox announces with each option.
    function group(scope, q) {
        const box = document.createElement("li");
        box.setAttribute("role", "presentation");
        const head = Object.assign(document.createElement("div"), {
            id: `location-group-${scope.id}`, className: "group-label", textContent: scope.name,
        });
        head.setAttribute("role", "presentation");
        const options = document.createElement("ul");
        options.setAttribute("role", "group");
        options.setAttribute("aria-labelledby", head.id);
        options.append(...shown.flatMap((p, i) => (p?.scope === scope.id ? [option(p, i, q)] : [])));
        box.append(head, options);
        return box;
    }

    // A listbox may own only options, so the empty message is one that cannot be chosen.
    function noMatch() {
        const item = Object.assign(document.createElement("li"), { className: "empty", textContent: "No matching location" });
        item.setAttribute("role", "option");
        item.setAttribute("aria-disabled", "true");
        return item;
    }

    // Shown first, and only once a place is chosen, the way back to none; it is shown[0], null.
    function clearOption() {
        const item = Object.assign(document.createElement("li"), {
            id: "location-option-0", className: "clear", textContent: "Select a location",
        });
        item.setAttribute("role", "option");
        item.setAttribute("aria-selected", "false");
        item.addEventListener("mousedown", (e) => e.preventDefault());
        item.addEventListener("click", () => pick(null));
        return item;
    }

    function render() {
        const q = key(location.value);
        const filtering = q !== "" && !byKey.has(q);
        const found = matchPlaces(places, location.value, (k) => byKey.has(k));
        shown = chosen && !filtering ? [null, ...found] : found;
        const cities = scopes.filter((s) => found.some((p) => p.scope === s.id));
        const groups = cities.map((s) => group(s, filtering ? q : ""));
        list.replaceChildren(...(found.length === 0 ? [noMatch()] : shown[0] === null ? [clearOption(), ...groups] : groups));
        setOpen(true);
    }

    function highlight(index) {
        list.querySelector('[aria-selected="true"]')?.setAttribute("aria-selected", "false");
        active = index;
        const item = byId(`location-option-${index}`);
        if (!item) return;
        item.setAttribute("aria-selected", "true");
        location.setAttribute("aria-activedescendant", item.id);
        item.scrollIntoView({ block: "nearest" });
    }

    function openList() {
        render();
        highlight(shown.indexOf(byKey.get(key(location.value))));
    }

    function pick(place) {
        setOpen(false);
        if (place) { touched.location = true; return choose(place); }
        chosen = null;
        location.value = "";
        touched.location = false;
        limitChoices();
        return update();
    }

    function onKey(e) {
        const open = !list.hidden;
        if (e.key === "ArrowDown" && (e.altKey || !open)) { e.preventDefault(); openList(); return; }
        if ((e.key === "ArrowDown" || e.key === "ArrowUp") && shown.length > 0) {
            e.preventDefault();
            highlight((active + (e.key === "ArrowDown" ? 1 : -1) + shown.length) % shown.length);
        } else if (e.key === "Enter" && open && shown.length > 0) {
            e.preventDefault();
            pick(active >= 0 ? shown[active] : shown.find((p) => p)); // unhighlighted, the first place
        } else if (e.key === "Escape" && open) {
            e.preventDefault();
            location.value = chosen ? chosen.label : ""; // back to the last place chosen
            setOpen(false);
            update();
        }
    }

    location.addEventListener("input", () => { render(); update(); });
    location.addEventListener("keydown", onKey);
    location.addEventListener("focus", () => location.select()); // typing replaces the name
    location.addEventListener("click", () => {
        if (!list.hidden) return;
        openList();
        location.select(); // a mouse click can undo the selection made on focus
    });
    toggle.addEventListener("mousedown", (e) => e.preventDefault()); // keep focus in the input
    toggle.addEventListener("click", () => {
        if (list.hidden) openList(); else setOpen(false);
        location.focus();
    });
    location.addEventListener("blur", () => {
        setOpen(false);
        const exact = byKey.get(key(location.value));
        touched.location = location.value.trim() !== "";
        if (exact && exact !== chosen) return choose(exact);
        if (exact) location.value = exact.label;
        return update();
    });

    // ---- the rest of the form, then the answers a link carries
    const upTo = (n) => Array.from({ length: n }, (_, i) => i + 1);
    addSegments("bhk", upTo(CHOICES));
    addSegments("bath", upTo(CHOICES + 1));
    byId("sqft").addEventListener("blur", () => { touched.sqft = true; update(); });
    byId("form").addEventListener("input", (e) => {
        if (e.target === location) return;
        if (e.target.name === "bhk") limitChoices();
        update();
    });
    byId("form").addEventListener("submit", (e) => e.preventDefault());
    byId("cities").textContent = scopes.map((s) => s.name).join(", ");

    const asked = new URLSearchParams(window.location.search);
    for (const name of ["bhk", "bath"]) {
        const input = byId(`${name}-${asked.get(name)}`);
        if (input?.name === name) input.checked = true;
    }
    const askedArea = parseArea(asked.get("sqft") ?? "");
    if (Number.isFinite(askedArea)) byId("sqft").value = String(askedArea);
    const place = places.find((p) => p.scope === asked.get("city") && p.name === asked.get("location"));
    if (place) return choose(place);
    limitChoices();
    return update();
}

if (typeof document !== "undefined") {
    start();
} else if (typeof module !== "undefined") {
    module.exports = { loadModel, readScopes, placesOf, matchPlaces, parseArea, money, span }; // tests/page_parity.js
}
