# Modern Algorithmic Game Theory (NOPT021)

This repository contains materials for the Modern Algorithmic Game Theory course taught at the Faculty of Mathematics and Physics at Charles University in Prague. You can find further information about the course on the associated webpage <https://sites.google.com/view/agtg-101>.

## Lecture Slides

You can find lecture slides for each lecture in the `slides` directory. If you find any issues or have suggestions for improvements, please let us know.

## Homework Assignments

Homework assignments for the course are available in the `tasks` directory in the form of Markdown files. You can find templates defining function names and signatures in the `templates` directory.

## Automated Tests

Automated tests for the homework assignments are located in the `tests` directory. The tests use the `pytest` package and compare outputs of your solutions against expected outputs produced by our reference implementations. You can execute all tests using the `pytest` command or run tests for individual weeks using `pytest tests/<file_name>.py`.

This way of testing is completely new this year, so if you encounter any issues, such as tolerance problems with floating-point comparisons or possibly incorrect expected outputs, please let us know on Discord.

### Selecting Individual Test Cases

Tests share a single session-scoped NumPy RNG state. Changing the order of tests in any way (e.g., running a subset of tests, commenting some tests out or reordering them) will result in a possibly different random state being passed to each selected test case, which may lead to different random games and strategies being generated than the ones used by the reference solution.

The ideal way to test your solutions during development is to use small, self-contained examples that you can implement easily and solve on paper, if needed, and use the provided tests only for final verification. However, if you still want to run the tests during development, it is possible to comment out the calls to `ndarrays_regression.check` in each test case as this will not affect the random state.

## Contact

For any questions or clarifications related to the course materials, do not hesitate to contact us. You can reach us through the course Discord server or the faculty emails.
Alternatively, if you spot typos or bugs in the course materials or have any suggestions how we could improve the course, feel free to reach out to us or submit a pull request yourself.
