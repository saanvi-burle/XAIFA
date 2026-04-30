import { useState, useEffect } from 'react'
import axios from 'axios'

interface Failure {
  failure_id: string
  sample_id: string
  source_path: string
  true_label: string
  predicted_label: string
  confidence: number
}

interface Run {
  run_id: string
  failures: Failure[]
}

function Failures() {
  const [runs, setRuns] = useState<Run[]>([])
  const [selectedRun, setSelectedRun] = useState<string>('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios.get('/api/runs')
      .then(res => {
        setRuns(res.data)
        if (res.data.length > 0) {
          setSelectedRun(res.data[0].run_id)
        }
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const currentRun = runs.find(r => r.run_id === selectedRun)

  return (
    <div>
      <h2 style={{ marginBottom: '1.5rem' }}>Failure Analysis</h2>
      
      <div className="card">
        <h3>Select Analysis Run</h3>
        <div className="form-group">
          <select 
            value={selectedRun}
            onChange={e => setSelectedRun(e.target.value)}
            style={{ maxWidth: '400px' }}
          >
            {runs.map(run => (
              <option key={run.run_id} value={run.run_id}>
                {run.run_id.slice(0, 8)}... ({run.failures.length} failures)
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading ? (
        <div className="loading">
          <div className="spinner"></div>
        </div>
      ) : currentRun ? (
        <div className="card">
          <h3>Failed Predictions ({currentRun.failures.length} cases)</h3>
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
              {currentRun.failures.map(failure => (
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
      ) : (
        <div className="card">
          <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '2rem' }}>
            No failures to display. Run an analysis first.
          </p>
        </div>
      )}

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