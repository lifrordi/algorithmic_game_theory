# Modern Algorithmic Game Theory (NOPT021)

This repository contains materials for the Modern Algorithmic Game Theory course taught at the Faculty of Mathematics and Physics at Charles University in Prague. You can find further information about the course on the associated webpage <https://sites.google.com/view/agtg-101>.

## Lecture Slides

You can find lecture slides for each lecture in the `slides` directory. If you find any issues or have suggestions for improvements, please let us know.

## Homework Assignments

Homework assignments for the course are available in the `tasks` directory in the form of Markdown files. You can find templates defining function names and signatures in the `templates` directory.

## Automated Tests

Automated tests for the homework assignments are located in the `tests` directory. The tests use the `pytest` package and compare outputs of your solutions against expected outputs produced by our reference implementations. You can execute all tests using the `pytest` command or run tests for individual weeks using `pytest tests/<file_name>.py`.

This way of testing is completely new this year, so if you encounter any issues, such as tolerance problems with floating-point comparisons or possibly incorrect expected outputs, please let us know on Discord.

### Selective test runs and RNG

Tests share a session-scoped NumPy RNG, so running subsets or reordering tests changes the random stream and can break things. Prefer running full files (e.g., `pytest tests/test_week01.py`).

## Contact

For any questions or clarifications related to the course materials, do not hesitate to contact us. You can reach us through the course Discord server or the faculty emails.
Alternatively, if you spot typos or bugs in the course materials or have any suggestions how we could improve the course, feel free to reach out to us or submit a pull request yourself.
