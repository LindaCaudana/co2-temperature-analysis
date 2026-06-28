# CO2 Temperature Analysis

Data mining project analyzing the relationship between global CO2 emissions and temperature change, using linear regression, clustering, and correlation analysis.

> Final project for **Tópicos Selectos de Ciencias de la Ingeniería 3** — UANL, FIME.

## 📋 Project Overview

This project applies data mining techniques to a real-world dataset to study how CO2 emissions relate to global temperature change over time, and how factors like GDP and population play a role.

**Problem statement:** Is there a measurable relationship between CO2 emissions per capita and the temperature change attributable to greenhouse gases, across countries and over time?

- **Dependent variable:** `temperature_change_from_co2` (or `temperature_change_from_ghg`)
- **Independent variables:** `co2_per_capita`, `gdp`, `population`, `year`

## 📊 Dataset

**Source:** [Our World in Data — CO2 and Greenhouse Gas Emissions](https://github.com/owid/co2-data)

- One row per country per year
- Covers data from 1750 to present
- Includes CO2 emissions, GDP, population, and temperature change attributable to greenhouse gases

Loaded directly in Python with:

```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv")
```

## 🎯 Objectives

1. Clean and explore the dataset (handle nulls, duplicates, inconsistencies).
2. Apply data mining techniques: linear regression, clustering, and correlation analysis.
3. Interpret results in the context of global climate change.
4. Communicate findings through visualizations and a final report.

## 🛠️ Tools

- **Language:** Python
- **Libraries:** pandas, numpy, matplotlib / seaborn, scikit-learn

## 📁 Repository Structure

```
co2-temperature-analysis/
├── data/               # Raw and processed datasets
├── notebooks/          # Jupyter notebooks (EDA, modeling, visualization)
├── src/                # Python scripts
├── results/            # Plots, exported tables, model outputs
├── README.md
└── requirements.txt
```

## 🚀 Project Tasks

- [ ] Dataset selection & description
- [ ] Data cleaning & exploratory data analysis (EDA)
- [ ] Correlation analysis
- [ ] Linear regression
- [ ] Clustering
- [ ] Data visualization
- [ ] Results interpretation
- [ ] Final report & presentation

## 📚 References

- Our World in Data. (2025). *CO2 and Greenhouse Gas Emissions* [Dataset]. https://github.com/owid/co2-data
- Global Carbon Project. *Global Carbon Budget*. https://globalcarbonbudget.org/

## 👤 Author

Proyecto Final — Tópicos Selectos de Ciencias de la Ingeniería 3
M.A. Cesar Yael Santillan Marroquin — UANL, FIME
