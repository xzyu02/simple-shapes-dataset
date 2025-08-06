# FOL-Structured Caption Generation for Multi-Shape Datasets

This document explains the First-Order Logic (FOL) structured caption generation system designed for studying predicate-argument structure in semantic understanding.

## Overview

The new `composer_multi.py` module generates captions that explicitly encode semantic relationships in First-Order Logic format, making it ideal for studying how well models capture "predicate-argument structure" - the elementary building blocks of semantic meaning.

## FOL Predicate Types

### 1. One-Place Predicates (Properties)

These describe intrinsic properties of individual objects:

- **Shape predicates**: `Circle(x)`, `Square(x)`, `Triangle(x)`, `Diamond(x)`, `Star(x)`, `Heart(x)`
- **Color predicates**: `Red(x)`, `Blue(x)`, `Green(x)`, `Yellow(x)`, etc.
- **Size predicates**: `Large(x)`, `Small(x)`, `Medium(x)`, `Tiny(x)`, `Huge(x)`
  - **Canvas-Aware Sizing**: Size descriptions are relative to canvas dimensions
  - 32x32 canvas: size 14 → "large" (44% of canvas)
  - 224x224 canvas: size 14 → "tiny" (6% of canvas)
  - Thresholds: tiny (<15%), small (<25%), medium (<35%), large (<45%), huge (≥45%)

**Example Caption**: "The image contains a tan circle and a green square."
**FOL Representation**: `∃x∃y(Circle(x) ∧ Tan(x) ∧ Square(y) ∧ Green(y))`

### 2. Two-Place Predicates (Relations)

These describe spatial and comparative relationships between objects:

- **Spatial relations**: `Near(x,y)`, `Above(x,y)`, `Below(x,y)`, `LeftOf(x,y)`, `RightOf(x,y)`
- **Distance relations**: `Far(x,y)`, `Close(x,y)`, `Adjacent(x,y)`

**Example Caption**: "A large, dark gray diamond is near a small, teal heart."
**FOL Representation**: `∃x∃y(Large(x) ∧ DarkGray(x) ∧ Diamond(x) ∧ Small(y) ∧ Teal(y) ∧ Heart(y) ∧ Near(x,y))`

### 3. Complex Noun Phrases with Multiple Modifiers

The system handles complex noun phrases that "unwind" multiple properties:

**Example**: "a large, red, circular object" encodes `Large(x) ∧ Red(x) ∧ Circle(x)`

This addresses the challenge described in semantic understanding research where complex modifiers must be correctly attributed to the same entity.

## Caption Generation Strategies

### 1. Simple Conjunction
For basic multi-object descriptions without explicit relationships:
- Template: `∃x∃y(P₁(x) ∧ P₂(x) ∧ Q₁(y) ∧ Q₂(y))`
- Example: "The image contains a red circle and a blue square."

### 2. Spatial Relationship
For explicit spatial relationships between objects:
- Template: `∃x∃y(P₁(x) ∧ P₂(x) ∧ Q₁(y) ∧ Q₂(y) ∧ R(x,y))`
- Example: "A large diamond is above a small heart."

### 3. Comparative Description
For size and property comparisons:
- Template: Emphasizes comparative predicates and size relationships
- Example: "The image contains different sized shapes: a large circle and a small square."

### 4. Positional Description
For overall spatial arrangements (3+ objects):
- Template: Describes global arrangement patterns
- Example: "A red circle, blue square, and green triangle are scattered across the canvas."

## Semantic Challenges Addressed

### 1. Multiple Modifiers
The system correctly handles complex noun phrases with multiple adjectives:
- **Challenge**: "an old, green, dirty car" must assign all three properties to the same object
- **Our approach**: "a large, dark gray diamond" → `Large(x) ∧ DarkGray(x) ∧ Diamond(x)`

### 2. Relational Predicates
Two-place predicates capture spatial and comparative relationships:
- **Challenge**: Understanding relationships between objects
- **Our approach**: "X is near Y" → `Near(x,y)`

### 3. Canvas-Aware Size Classification
Size descriptions adapt to canvas dimensions for semantic consistency:
- **Challenge**: Absolute size values have different semantic meaning on different canvas sizes
- **Our approach**: Size 14 on 32x32 canvas → "large" vs. size 14 on 224x224 canvas → "tiny"
- **Benefit**: Ensures size predicates have consistent semantic meaning across different image resolutions

### 4. Quantification and Scope
Proper use of existential quantification for multiple objects:
- **Challenge**: Correct scoping of quantifiers
- **Our approach**: `∃x∃y(...)` for two-object relationships

## Usage in Research

This caption generation system is designed to support research into:

1. **Compositional Understanding**: How well do models understand that properties combine to describe single objects?

2. **Relational Reasoning**: Can models correctly identify and reason about spatial relationships?

3. **Semantic Parsing**: How accurately can models map natural language to logical representations?

4. **Predicate-Argument Structure**: Do models understand the fundamental semantic building blocks?

## Example Generated Captions

### Two-Shape Canvas
- Simple: "The image contains a red circle and a blue square."
- Spatial: "A red circle is near a blue square."
- Comparative: "The image contains different sized shapes: a large red circle and a small blue square."

### Three-Shape Canvas
- Conjunction: "There are a red triangle, a green circle, and a blue square."
- Positional: "A red triangle, green circle, and blue square are arranged linearly."
- Complex: "A large red triangle is above a small green circle near a blue square."

## Integration with Existing Pipeline

The new system integrates seamlessly with the existing BERT embedding pipeline:
1. Generate FOL-structured captions
2. Create BERT embeddings from natural language captions
3. Study how embeddings capture logical structure

This allows researchers to investigate whether language models' internal representations preserve the compositional semantic structure encoded in the captions.
