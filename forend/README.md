t# AttendIQ: Student Attendance Analytics & Absenteeism Prediction System

A working prototype that analyses student attendance, predicts who is likely to finish the semester below 75%, and recommends corrective actions. It runs entirely in the browser with HTML, CSS and JavaScript. No server or installation is needed.

## Problem statement

Attendance records collected across semesters, subjects and sessions are a valuable data source. This project builds a Big Data Analytics and Machine Learning solution that:

- Analyses subject-wise and semester-wise attendance patterns
- Identifies chronic absenteeism and irregular trends
- Predicts future attendance shortage for students
- Correlates attendance with academic performance
- Presents insights through a dashboard

**Pipeline:** Raw Data → Preprocessing → Attendance Analytics → ML Prediction → Alert → Recommendation

## Features

| Deliverable | Where it is in the project |
|---|---|
| Dataset (cleaned and processed) | Sections 1 and 2 of `script.js` |
| Analytics (attendance trend insights) | Analytics tab |
| ML model (shortage prediction) | Section 4 of `script.js` |
| Evaluation (ML performance metrics) | Prediction tab |
| Dashboard | All tabs (Overview, Analytics, Prediction, Students) |
| Recommendation (corrective actions) | Students tab and Alerts on the Overview tab |
| Source code | This folder |

## Project structure

```
attendance-system/
├── index.html   # page layout: sidebar and four tabs
├── style.css    # styling
├── script.js    # data, preprocessing, analytics, ML model, dashboard logic
└── README.md
```

## How to run

1. Keep all files in the same folder.
2. Open `index.html` in a modern browser (Chrome, Edge or Firefox).
3. An internet connection is needed once, because Chart.js is loaded from a CDN.

The ML model trains automatically every time the page loads.

## How it works

### 1. Raw data
The prototype generates a synthetic dataset of 120 students across 3 semesters. Each record has 15 weekly attendance rates and internal marks. A few missing and invalid values are added on purpose to demonstrate cleaning.

### 2. Preprocessing
- Missing weekly values are filled with the student's own average for that semester.
- Out-of-range values are clipped to the 0–100% range.
- Features are built from week 9 data only, so the model predicts early, as it would in real use:
  - attendance so far
  - attendance in the last 3 weeks
  - trend (recent attendance minus early attendance)
  - internal marks
  - previous semester attendance
  - number of weeks below 60%

### 3. Attendance analytics
- Subject-wise attendance
- Semester-wise average attendance
- Weekly trend by student group
- Pearson correlation between attendance and marks

### 4. ML prediction
- **Model:** Logistic regression written in plain JavaScript, trained with gradient descent.
- **Target:** Whether a student's final attendance will be below 75%.
- **Training and testing:** Trained on Semesters 1 and 2, tested on Semester 3.
- **Metrics:** Accuracy, precision, recall, F1 score, confusion matrix and feature weights.

### 5. Alerts
Each student gets a shortage probability and a risk level:

| Probability | Level |
|---|---|
| 70% or more | High |
| 40% to 70% | Medium |
| Below 40% | Low |

### 6. Recommendations
Each student gets actions based on their data, for example:
- How much of the remaining classes they must attend to reach 75%
- A mentor meeting if attendance is falling
- Counselling and parent notification if attendance is below 65%
- Remedial classes if marks are below 50
- A condonation or medical-leave review if 75% can no longer be reached

## Configuration

All settings are at the top of `script.js` or in the ML section:

| Setting | Default | Meaning |
|---|---|---|
| `CUT` | `.75` | Attendance shortage threshold |
| `WEEK_NOW` | `9` | Week at which predictions are made |
| `WEEKS` | `15` | Weeks in a semester |
| `N` | `120` | Number of synthetic students |
| Training loop | `1000`, `.1` | Iterations and learning rate |

## Using your own dataset

Replace the "RAW DATA" block in `script.js` with your records in this shape:

```js
{ id: 1, group: 'A', sem: 1, weekly: [0.9, 0.8, ...15 values], marks: 72 }
```

Reload the page and the model retrains on your data.

## Try a single prediction

Open the browser console (F12) and run:

```js
prob({ f: { att: .62, recent: .55, slope: -.10, marks: 45, prev: .70, low: 3 } })
```

It returns the shortage probability between 0 and 1.

## Limitations

- The default dataset is synthetic, so the model scores will look higher than they would on real data.
- Logistic regression is a simple baseline. For larger datasets, a random forest or gradient boosting model trained in Python could be added.
- Subject-wise attendance in the demo is derived from overall attendance with small offsets per subject.

## Tech stack

- HTML, CSS, JavaScript (no framework)
- [Chart.js](https://www.chartjs.org/) for charts

## Future improvements

- CSV upload for real attendance data
- A Python backend (scikit-learn) with exported predictions
- Email or SMS alerts to students and mentors
- A student-facing page to check personal attendance
