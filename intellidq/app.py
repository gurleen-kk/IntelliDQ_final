import os
import json
import uuid
import traceback
import requests
from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import io
import warnings
warnings.filterwarnings('ignore')

FOUNDRY_API_KEY = ""
FOUNDRY_ENDPOINT = ""

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def run_dq_checks(df):
    results = {}
    total_rows = len(df)
    total_cells = total_rows * len(df.columns)

    missing = df.isnull().sum()
    missing_pct = (missing / total_rows * 100).round(2)
    completeness_score = round(100 - (missing.sum() / total_cells * 100), 2)
    results['completeness'] = {
        'score': completeness_score,
        'total_missing_cells': int(missing.sum()),
        'columns': {
            col: {'missing': int(missing[col]), 'pct': float(missing_pct[col])}
            for col in df.columns if missing[col] > 0
        }
    }

    dup_rows = df.duplicated().sum()
    uniqueness_score = round((1 - dup_rows / max(total_rows, 1)) * 100, 2)
    results['uniqueness'] = {
        'score': uniqueness_score,
        'duplicate_rows': int(dup_rows),
        'duplicate_pct': round(dup_rows / max(total_rows, 1) * 100, 2)
    }

    validity_issues = {}
    for col in df.columns:
        col_issues = []
        series = df[col].dropna()

        if df[col].dtype == object:
            numeric_count = pd.to_numeric(series, errors='coerce').notna().sum()
            if 0 < numeric_count < len(series):
                col_issues.append(f'Mixed types: {numeric_count} numeric-looking values in text column')

        if pd.api.types.is_numeric_dtype(df[col]):
            neg_count = (df[col] < 0).sum()
            name_lower = col.lower()
            if neg_count > 0 and any(kw in name_lower for kw in ['age', 'count', 'quantity', 'amount', 'price', 'cost', 'id']):
                col_issues.append(f'{neg_count} negative values (potentially invalid for "{col}")')

        if col_issues:
            validity_issues[col] = col_issues

    validity_score = round((1 - len(validity_issues) / max(len(df.columns), 1)) * 100, 2)
    results['validity'] = {
        'score': validity_score,
        'issues': validity_issues
    }

    consistency_issues = {}
    for col in df.columns:
        if df[col].dtype == object:
            series = df[col].dropna().astype(str)
            lower_vals = series.str.lower().value_counts()
            orig_vals = series.value_counts()
            if len(lower_vals) < len(orig_vals):
                diff = len(orig_vals) - len(lower_vals)
                consistency_issues[col] = f'{diff} case-variant entries (e.g. "Yes" vs "yes")'

    consistency_score = round((1 - len(consistency_issues) / max(len(df.columns), 1)) * 100, 2)
    results['consistency'] = {
        'score': consistency_score,
        'issues': consistency_issues
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    anomaly_rows = []
    anomaly_indices = []

    if len(numeric_cols) >= 1 and total_rows >= 10:
        X = df[numeric_cols].fillna(df[numeric_cols].median())
        contamination = min(0.1, max(0.01, 1.0 / np.sqrt(total_rows)))
        clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        preds = clf.fit_predict(X)
        scores = clf.score_samples(X)
        anomaly_indices = list(np.where(preds == -1)[0])
        if anomaly_indices:
            sample = df.iloc[anomaly_indices[:20]].copy()
            sample['_anomaly_score'] = -scores[anomaly_indices[:20]]
            sample['_row_index'] = anomaly_indices[:20]
            anomaly_rows = sample.head(20).to_dict('records')

    results['anomalies'] = {
        'count': len(anomaly_indices),
        'pct': round(len(anomaly_indices) / max(total_rows, 1) * 100, 2),
        'numeric_cols_used': numeric_cols,
        'sample_rows': anomaly_rows[:10]
    }

    anomaly_score = round((1 - results['anomalies']['pct'] / 100) * 100, 2)
    overall = (
        0.3 * results['completeness']['score'] +
        0.2 * results['uniqueness']['score'] +
        0.2 * results['validity']['score'] +
        0.15 * results['consistency']['score'] +
        0.15 * anomaly_score
    )
    results['overall_score'] = round(overall, 2)
    results['anomaly_score'] = anomaly_score

    results['meta'] = {
        'rows': total_rows,
        'columns': len(df.columns),
        'column_names': list(df.columns),
        'dtypes': {col: str(df[col].dtype) for col in df.columns},
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
    }

    col_stats = {}
    for col in df.columns:
        stat = {
            'dtype': str(df[col].dtype),
            'missing': int(df[col].isnull().sum()),
            'missing_pct': round(df[col].isnull().sum() / total_rows * 100, 2),
            'unique': int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            stat.update({
                'min': round(float(df[col].min()), 4) if not pd.isna(df[col].min()) else None,
                'max': round(float(df[col].max()), 4) if not pd.isna(df[col].max()) else None,
                'mean': round(float(df[col].mean()), 4) if not pd.isna(df[col].mean()) else None,
                'std': round(float(df[col].std()), 4) if not pd.isna(df[col].std()) else None,
            })
        else:
            vc = df[col].value_counts()
            stat['top_values'] = {str(k): int(v) for k, v in vc.head(5).items()}
        col_stats[col] = stat
    results['column_stats'] = col_stats

    return results


def get_foundry_summary(dq_results):
    try:
        prompt = f"""
        You are a data quality analyst writing a summary for a non-technical business stakeholder.
        Avoid technical jargon. Do not mention column names, code, or statistical terms directly.
        Write in clear, professional English as if presenting findings to a senior manager or decision-maker.
        Focus on the business impact of the issues found and what needs to happen next.

        Structure your response exactly as follows:

        **Headline:** A single sentence summarising the overall state of the data.

        **Summary:** A short paragraph explaining what was found in plain English.

        **Recommended Actions:**
        For each action, prefix it with a severity tag using this scale:
        [Critical] - must be fixed immediately before the data can be used
        [High] - significant issue that will affect reporting accuracy
        [Moderate] - should be addressed soon to maintain data reliability
        [Low] - minor issue, fix when possible
        [Review] - needs manual investigation to determine next steps

        List 3-5 actions, each on its own line, starting with the severity tag in square brackets.
        Order them from highest to lowest severity so stakeholders know what to prioritise first.

        Here are the data quality findings:
        Overall data quality score: {dq_results['overall_score']}/100
        Completeness: {dq_results['completeness']['score']}% — {dq_results['completeness']['total_missing_cells']} missing values found
        Uniqueness: {dq_results['uniqueness']['score']}% — {dq_results['uniqueness']['duplicate_rows']} duplicate records found
        Anomalies detected: {dq_results['anomalies']['count']} unusual records ({dq_results['anomalies']['pct']}% of dataset)
        Number of fields with validity issues: {len(dq_results['validity']['issues'])}
        Number of fields with consistency issues: {len(dq_results['consistency']['issues'])}
        """
        response = requests.post(
            FOUNDRY_ENDPOINT,
            headers={"Authorization": f"Bearer {FOUNDRY_API_KEY}", "Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": prompt}]},
            timeout=15
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'Empty filename'}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ('.csv', '.xlsx', '.xls'):
        return jsonify({'error': 'Only CSV and Excel files are supported'}), 400

    try:
        if ext == '.csv':
            df = pd.read_csv(f)
        else:
            df = pd.read_excel(f)

        if df.empty:
            return jsonify({'error': 'File is empty or could not be parsed'}), 400

        results = run_dq_checks(df)
        results['filename'] = f.filename

        foundry_summary = get_foundry_summary(results)
        if foundry_summary:
            results['ai_summary'] = foundry_summary

        uid = str(uuid.uuid4())[:8]
        path = os.path.join(UPLOAD_FOLDER, f'{uid}.json')
        with open(path, 'w') as fp:
            json.dump(results, fp, default=str)
        results['_uid'] = uid

        results = json.loads(json.dumps(results, default=str).replace(': NaN', ': null').replace(':NaN', ':null'))
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/export/csv/<uid>')
def export_csv(uid):
    path = os.path.join(UPLOAD_FOLDER, f'{uid}.json')
    if not os.path.exists(path):
        return 'Not found', 404
    with open(path) as f:
        data = json.load(f)

    rows = []
    for col, stat in data['column_stats'].items():
        rows.append({
            'Column': col,
            'Type': stat['dtype'],
            'Missing': stat['missing'],
            'Missing %': stat['missing_pct'],
            'Unique Values': stat['unique'],
            'Min': stat.get('min', ''),
            'Max': stat.get('max', ''),
            'Mean': stat.get('mean', ''),
            'Std Dev': stat.get('std', ''),
        })

    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='intellidq_report.csv'
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)