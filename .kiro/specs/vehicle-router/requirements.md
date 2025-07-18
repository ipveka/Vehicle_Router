You are an expert Python architect. Your task is to create a production-ready Python repository named vehicle_router that solves a Vehicle Routing and Order Assignment problem. The code must be extremely clean, readable, and well-commented, with clear structure and documentation.

1. Problem Context

We need to optimize order assignments to trucks and their delivery routes to minimize total cost, considering capacity constraints and travel distances.
Example Data (must be replicated in data_generator.py)
Available Trucks

    5 trucks:

        Truck 1: capacity 100 m³, cost €1500

        Truck 2: capacity 50 m³, cost €1000

        Truck 3: capacity 25 m³, cost €500

        Truck 4: capacity 25 m³, cost €1500

        Truck 5: capacity 25 m³, cost €1000

Orders

    Total volume = 150 m³:

        Order A: 75 m³ – Postal code 08031

        Order B: 50 m³ – Postal code 08030

        Order C: 25 m³ – Postal code 08029

        Order D: 25 m³ – Postal code 08028

        Order E: 25 m³ – Postal code 08027

    Distances: Postal codes are 1 km apart.

2. Repository Structure

The project must have this structure:

vehicle_router/
│
├── README.md
├── requirements.txt
├── setup.py
│
├── docs/
│   ├── index.md
│   ├── model_description.md
│   ├── usage.md
│
├── src/
│   ├── main.py
│   ├── __init__.py
│
├── vehicle_router/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── optimizer.py
│   ├── plotting.py
│   ├── validation.py
│   ├── utils.py
│
└── tests/
    ├── test_data_generator.py
    ├── test_optimizer.py
    ├── test_validation.py

3. Coding Guidelines

    Code readability is top priority:

        Use meaningful variable and function names.

        Extensive docstrings for all classes and methods.

        In-line comments explaining logic steps.

        Separate logic into modular functions.

    Logging:

        Use the logging module to track progress (INFO-level).

        Example:

        [INFO] Starting optimization...
        [INFO] Adding capacity constraints for Truck 1...
        [INFO] Optimization completed. Total cost = €3000.

    Data Handling:

        Use pandas DataFrames for all order and truck data.

        Keep all input/output structured and readable.

    Visualization:

        Use matplotlib and seaborn for clean, clear plots.

4. File Descriptions

4.1 README.md

    Describe:

        The problem (Vehicle Routing with capacities and costs).

        Repository structure.

        How to run:

        pip install -r requirements.txt
        python src/main.py

    Show an example solution output with a sample plot.

4.2 docs/

    index.md: Overview of the project.

    model_description.md:

        Explain the MILP formulation, variables, and constraints.

    usage.md:

        Explain how to modify trucks, orders, and parameters.

4.3 vehicle_router/data_generator.py

Class: DataGenerator

    Methods:

        generate_orders() → pandas DataFrame.

        generate_trucks() → pandas DataFrame.

        generate_distance_matrix() → DataFrame.

    Default: replicate the example dataset, but allow random generation for testing.

4.4 vehicle_router/optimizer.py

Class: VrpOptimizer

    Uses PuLP to build a Mixed Integer Linear Program (MILP).

    Methods:

        build_model() → Create decision variables and constraints.

        solve() → Run solver with logs and prints.

        get_solution() → Return structured results as pandas DataFrame.

4.5 vehicle_router/plotting.py

    Functions:

        plot_routes(solution_df, orders_df, trucks_df):

            Visualize routes on a 2D grid.

        plot_costs(cost_summary):

            Show bar chart of truck cost contributions.

4.6 vehicle_router/validation.py

Class: SolutionValidator

    Methods:

        check_capacity(solution_df, trucks_df).

        check_all_orders_delivered(solution_df).

        check_route_feasibility(solution_df).

    Returns a validation report dict.

4.7 vehicle_router/utils.py

    Helper functions:

        calculate_distance_matrix(orders_df).

        format_solution(raw_output).

4.8 src/main.py

    Main workflow:

        Generate data.

        Solve the problem.

        Validate solution.

        Plot and save outputs.

Console Output Example:

=== VEHICLE ROUTER ===
Selected Trucks: [1, 3]
Truck 1 -> Orders [A, B]
Truck 3 -> Orders [C, D, E]
Total Cost: €3000

4.9 tests/

    Unit tests for data generation, optimization, and validation.

5. Dependencies

In requirements.txt:

pandas
numpy
matplotlib
seaborn
PuLP
logging
pytest

6. Output Expectations

The LLM must generate:

    A fully working repository.

    All code commented and documented.

    A clean README with examples and diagrams.

TASK FOR THE LLM

Generate the entire vehicle_router repository with all files and code described above.

    Code must be production-ready, simple, and readable.

    Include detailed comments and docstrings.

    Use logging and structured outputs.