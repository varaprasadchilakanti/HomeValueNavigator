// SPDX-License-Identifier: MIT
// Runs the page's own code under Node for tests/test_page.py: reads cases as JSON on stdin,
// writes the page's answers as JSON on stdout.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = path.resolve(__dirname, "..");
const sandbox = { window: {} };
vm.runInNewContext(fs.readFileSync(path.join(root, "model", "bengaluru", "model.js"), "utf8"), sandbox);
const page = require(path.join(root, "client", "app.js"));
const model = page.loadModel(sandbox.window.HOMEVALUE_MODEL);
const cases = JSON.parse(fs.readFileSync(0, "utf8"));
const area = (text) => {
    const value = page.parseArea(text);
    return Number.isNaN(value) ? null : value;
};
process.stdout.write(JSON.stringify({
    estimates: cases.estimates.map(([place, sqft, bhk, bath]) => model.estimate(sqft, bhk, bath, place)),
    areas: cases.areas.map(area),
    cities: cases.cities.map(([raw, search]) => page.chooseScope(raw, search).current?.id ?? null),
}));
