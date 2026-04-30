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

function Dashboard() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios.get('/api/runs')
      .then(res => {
        setRuns(res.data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const latestRun = runs[0]

  return (
    <div>
      <h2 style={{ marginBottom: '1.5rem' }}>Dashboard</h2>
      
      {loading ? (
        <div className="loading">
          <div className="spinner"></div>
        </div>
      ) : latestRun ? (
        <>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="value">{latestRun.total_samples}</div>
              <div className="label">Total Samples</div>
            </div>
            <div className="stat-card">
              <div className="value" style={{ color: 'var(--success)' }}>{latestRun.correct_predictions}</div>
              <div className="label">Correct</div>
            </div>
            <div className="stat-card">
              <div className="value" style={{ color: 'var(--danger)' }}>{latestRun.failed_predictions}</div>
              <div className="label">Failed</div>
            </div>
            <div className="stat-card">
              <div className="value">{(latestRun.accuracy * 100).toFixed(1)}%</div>
              <div className="label">Accuracy</div>
            </div>
          </div>

          <div className="card">
            <h2>Recent Analysis Runs</h2>
            <table className="table">
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>Model</th>
                  <th>Dataset</th>
                  <th>Accuracy</th>
                  <th>Date</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {runs.map(run => (
                  <tr key={run.run_id}>
                    <td>{run.run_id.slice(0, 8)}...</td>
                    <td>{run.model_id.slice(0, 8)}...</td>
                    <td>{run.dataset_id.slice(0, 8)}...</td>
                    <td>{(run.accuracy * 100).toFixed(1)}%</td>
                    <td>{new Date(run.created_at).toLocaleDateString()}</td>
                    <td>
                      <span className={`badge badge-${run.status === 'completed' ? 'success' : 'warning'}`}>
                        {run.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div className="card">
          <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '2rem' }}>
            No analysis runs yet. Upload a model and dataset to get started.
          </p>
        </div>
      )}
    </div>
  )
}

export default Dashboard