# Security

Report a vulnerability privately through GitHub's
[security advisories](https://github.com/varaprasadchilakanti/HomeValueNavigator/security/advisories/new),
not in a public issue. This is a one-person project, maintained when time allows; no reply
time is promised.

| Version | Supported |
|---|---|
| 0.2.x, and the latest commit on `main` | yes |
| 0.1.0 and anything older | no |

The page runs entirely in the browser and sends nothing it is given. It keeps the answers
after the `#` in the address, which browsers do not send to the server; a link someone
copies carries them. GitHub Pages, which
serves it, keeps its own access logs, as any web host does. The API reads one local JSON
file and stores nothing.
