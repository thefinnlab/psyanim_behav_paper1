# Psyanim 2.0 - Introduction and Overview

Psyanim 2.0 is a tool for rapid design, prototyping and deployment of psychology research experiments involving 2D procedural animation.

For a quick intro to Psyanim 2.0, check out our [docs](https://thefinnlab.github.io/psyanim-docs)!

To run the test suite:

- Run `npm i` from project root
- Start a watch service in a separate terminal with `npm run dev-watch`
- Start a local server to host the test suite with `npm run dev-serve`
- Navigate to `localhost:3000` in your web browser. Controls are at bottom of page.

To view the API docs from this repo:

- First build the docs with: `npm run docs-build`
- Then start a server to host the docs locally on port `5000` with: `npm run docs-serve`
- Then navigate to `localhost:5000` in your web browser

Psyanim 2.0 depends on [`psyanim-utils`](https://github.com/thefinnlab/psyanim-utils), which is a small collection of javascript APIs that can be used in *both* browsers and nodejs apps.