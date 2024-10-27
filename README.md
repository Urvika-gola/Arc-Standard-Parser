# Dependency Parsing with Rule-Based and Arc-Standard Parsers

## Overview
This project implements three dependency parsers:
1. **DummyParser**: A basic rule-based parser.
2. **LessDummyParser**: An improved rule-based parser with additional dependency rules.
3. **ArcStandardParser**: An Arc-Standard transition-based parser using left and right arc operations.

Each parser predicts the syntactic structure of sentences using various configurations of POS tags and dependency relations, handling shifts and arc operations to build dependency trees.

## Features

- **DummyParser**:
  - Tags verbs as root nodes and assigns dependency relations to other tokens based on fixed rules.
  
- **LessDummyParser**:
  - Improved rules for finding heads and dependencies based on POS patterns.
  - Custom rules for handling common syntactic patterns, such as noun and verb sequences.

- **ArcStandardParser**:
  - Arc-Standard parsing logic with `Shift`, `Left Arc`, and `Right Arc` operations.
  - Uses counts of transitions and POS tag configurations to predict dependencies.
  - Defaults to specific heuristics when encountering ambiguous configurations.

