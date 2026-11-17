// SPDX-License-Identifier: MIT
// Loaded in <head>, before the first paint, so a saved theme applies without a flash.
"use strict";
try {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
} catch { /* storage blocked: follow the system setting */ }
