import { useState, useEffect } from 'react'
import axios from 'axios'

interface RunSummary {
  run_id: string
  model_id: string
  dataset_id: string
  total_samples: number
  correct_predictions: number
  failed_predictions: number
  accuracy: number
  created_at: string
  status: string
}

interface Failure {
  failure_id: string
  sample_id: string
  source_path: string
  true_label: string
  predicted_label: string
  confidence: number
}

function Failures() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [selectedRun, setSelectedRun] = useState<string>('')
  const [failures, setFailures] = useState<Failure[]>([])
  const [loadingRuns, setLoadingRuns] = useState(true)
  const [loadingFailures, setLoadingFailures] = useState(false)

  // Load run list on mount
  useEffect(() => {
    axios.get('/api/runs')
      .then(res => {
        setRuns(res.data)
        if (res.data.length > 0) {
          setSelectedRun(res.data[0].run_id)
        }
        setLoadingRuns(false)
      })
      .catch(() => setLoadingRuns(false))
  }, [])

  // Load failures whenever selected run changes
  useEffect(() => {
    if (!selectedRun) {
      setFailures([])
      return
    }
    setLoadingFailures(true)
    axios.get(`/api/runs/${selectedRun}/failures`)
      .then(res => {
        setFailures(res.data)
        setLoadingFailures(false)
      })
      .catch(() => {
        setFailures([])
        setLoadingFailures(false)
      })
  }, [selectedRun])

  return (
    <div>
      <h2 style={{ marginBottom: '1.5rem' }}>Failure Analysis</h2>

      <div className="card">
        <h3>Select Analysis Run</h3>
        {loadingRuns ? (
          <div className="loading">Loading runs...</div>
        ) : runs.length === 0 ? (
          <p style={{ color: 'var(--text-muted)' }}>No runs yet. Run an analysis first.</p>
        ) : (
          <div className="form-group">
            <select
              value={selectedRun}
              onChange={e => setSelectedRun(e.target.value)}
              style={{ maxWidth: '400px' }}
            >
              {runs.map(run => (
                <option key={run.run_id} value={run.run_id}>
                  {run.run_id.slice(0, 8)}... ({run.failed_predictions} failures)
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {loadingFailures ? (
        <div className="loading">
          <div className="spinner"></div>
        </div>
      ) : failures.length > 0 ? (
        <div className="card">
          <h3>Failed Predictions ({failures.length} cases)</h3>
          <table className="table">
            <thead>
              <tr>
                <th>Failure ID</th>
                <th>True Label</th>
                <th>Predicted</th>
                <th>Confidence</th>
                <th>Source</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {failures.map(failure => (
                <tr key={failure.failure_id}>
                  <td>{failure.failure_id}</td>
                  <td>
                    <span className="badge badge-danger">{failure.true_label}</span>
                  </td>
                  <td>
                    <span className="badge badge-warning">{failure.predicted_label}</span>
                  </td>
                  <td>{(failure.confidence * 100).toFixed(1)}%</td>
                  <td style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {failure.source_path}
                  </td>
                  <td>
                    <button className="btn btn-secondary" style={{ padding: '0.375rem 0.75rem', fontSize: '0.75rem' }}>
                      View XAI
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : selectedRun ? (
        <div className="card">
          <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '2rem' }}>
            No failures recorded for this run.
          </p>
        </div>
      ) : null}

      <div className="card">
        <h3>XAI Explanation Methods</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
          <div style={{ padding: '1rem', background: 'var(--background)', borderRadius: '6px' }}>
            <h4 style={{ color: 'var(--primary)' }}>Grad-CAM</h4>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
              Gradient-weighted Class Activation Mapping - visualizes which regions influenced the prediction.
            </p>
          </div>
          <div style={{ padding: '1rem', background: 'var(--background)', borderRadius: '6px' }}>
            <h4 style={{ color: 'var(--primary)' }}>SHAP</h4>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
              SHapley Additive exPlanations - assigns importance values to each feature.
            </p>
          </div>
          <div style={{ padding: '1rem', background: 'var(--background)', borderRadius: '6px' }}>
            <h4 style={{ color: 'var(--primary)' }}>LIME</h4>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
              Local Interpretable Model-agnostic Explanations - explains individual predictions.
            </p>
          </div>
          <div style={{ padding: '1rem', background: 'var(--background)', borderRadius: '6px' }}>
            <h4 style={{ color: 'var(--primary)' }}>Fusion</h4>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
              Combined explanation from Grad-CAM, SHAP, and LIME methods.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Failures