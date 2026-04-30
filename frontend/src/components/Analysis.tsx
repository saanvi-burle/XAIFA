import { useState, useEffect } from 'react'
import axios from 'axios'

interface PipelineStep {
  order: number
  name: string
  status: string
}

function Analysis() {
  const [steps, setSteps] = useState<PipelineStep[]>([])
  const [loading, setLoading] = useState(true)
  const [modelId, setModelId] = useState('')
  const [datasetId, setDatasetId] = useState('')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<any>(null)

  useEffect(() => {
    axios.get('/api/runs/pipeline')
      .then(res => {
        setSteps(res.data.steps)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const runAnalysis = async () => {
    if (!modelId || !datasetId) return
    setRunning(true)
    try {
      const res = await axios.post('/api/runs/analyze', {
        model_id: modelId,
        dataset_id: datasetId,
      })
      setResult(res.data)
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Analysis failed')
    }
    setRunning(false)
  }

  return (
    <div>
      <h2 style={{ marginBottom: '1.5rem' }}>Run Analysis</h2>
      
      <div className="card">
        <h3>Start New Analysis</h3>
        <div className="form-group">
          <label>Model ID</label>
          <input 
            type="text" 
            value={modelId}
            onChange={e => setModelId(e.target.value)}
            placeholder="Enter model ID from upload"
          />
        </div>
        <div className="form-group">
          <label>Dataset ID</label>
          <input 
            type="text" 
            value={datasetId}
            onChange={e => setDatasetId(e.target.value)}
            placeholder="Enter dataset ID from upload"
          />
        </div>
        <button 
          className="btn btn-primary" 
          onClick={runAnalysis}
          disabled={!modelId || !datasetId || running}
        >
          {running ? 'Running Analysis...' : 'Run Analysis'}
        </button>
      </div>

      <div className="card">
        <h3>Analysis Pipeline</h3>
        {loading ? (
          <div className="loading">Loading...</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {steps.map(step => (
              <div 
                key={step.order} 
                style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '1rem',
                  padding: '0.75rem',
                  background: 'var(--background)',
                  borderRadius: '6px'
                }}
              >
                <span 
                  style={{ 
                    width: '24px', 
                    height: '24px', 
                    borderRadius: '50%', 
                    background: step.status === 'completed' ? 'var(--success)' : 'var(--border)',
                    color: 'white',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '0.75rem',
                    fontWeight: '600'
                  }}
                >
                  {step.order}
                </span>
                <span>{step.name}</span>
                <span 
                  className={`badge badge-${step.status === 'completed' ? 'success' : 'warning'}`}
                  style={{ marginLeft: 'auto' }}
                >
                  {step.status}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {result && (
        <div className="card">
          <h3>Analysis Results</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="value">{result.total_samples}</div>
              <div className="label">Total</div>
            </div>
            <div className="stat-card">
              <div className="value" style={{ color: 'var(--success)' }}>{result.correct_predictions}</div>
              <div className="label">Correct</div>
            </div>
            <div className="stat-card">
              <div className="value" style={{ color: 'var(--danger)' }}>{result.failed_predictions}</div>
              <div className="label">Failed</div>
            </div>
            <div className="stat-card">
              <div className="value">{(result.accuracy * 100).toFixed(1)}%</div>
              <div className="label">Accuracy</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Analysis