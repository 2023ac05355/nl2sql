📌 PROJECT PLAN (READ BEFORE WRITING CODE)

Goal
Build a modular, controllable Text-to-SQL (NL2SQL) system using the Spider dataset for an academic dissertation.

Constraints

Academic project, not a product

Clean, minimal, explainable code

Avoid over-engineering

No UI or APIs initially

Current State

spider_data/ exists and must NOT be modified

Target Folder Structure

project_root/
├── spider_data/
├── data/
│   ├── processed/
│   └── splits/
├── src/
│   ├── config/
│   ├── data/
│   ├── model/
│   ├── training/
│   └── inference/
├── experiments/
└── README.md


Development Order (IMPORTANT)

Data loading (Spider JSON → samples)

Preprocessing (SQL normalization, input formatting)

Base model wrapper (no training logic inside)

Simple inference pipeline

Training loop (basic)

Policy layer (future)

Coding Rules

One responsibility per file

Config-driven behavior (YAML)

Prefer clarity over cleverness

Functions > scripts

Naming Conventions

snake_case for files

Clear, academic naming (no slang)