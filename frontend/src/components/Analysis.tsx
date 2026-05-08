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

  const [showPipeline, setShowPipeline] =
    useState(true)

  const [csvData, setCsvData] = useState<
    string[][]
  >([])

  useEffect(() => {
    axios
      .get('/api/runs/pipeline')
      .then((res) => {
        setSteps(res.data.steps)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const loadCSV = async () => {
    try {
      const response = await fetch(
        `http://127.0.0.1:8001/results/final_results.csv?t=${Date.now()}`
      )

      const text = await response.text()

      const rows = text
        .trim()
        .split('\n')
        .map((row) => row.split(','))

      setCsvData(rows)

      console.log('CSV LOADED')
      console.log(rows)

    } catch (err) {
      console.error('CSV LOAD ERROR:', err)
    }
  }

  const runAnalysis = async () => {
    if (!modelId || !datasetId) return

    setRunning(true)

    setShowPipeline(true)

    try {
      const res = await axios.post(
        '/api/runs/analyze',
        {
          model_id: modelId,
          dataset_id: datasetId,
        }
      )

      console.log('===========================')
      console.log('FULL ANALYSIS RESPONSE')
      console.log(res.data)
      console.log('===========================')

      setResult(res.data)

      // LOAD CSV TABLE
      await loadCSV()

      setTimeout(() => {
        setShowPipeline(false)
      }, 1500)

    } catch (err: any) {
      alert(
        err.response?.data?.detail ||
          'Analysis failed'
      )
    }

    setRunning(false)
  }

  return (
    <div>
      <h2 style={{ marginBottom: '1.5rem' }}>
        Run Analysis
      </h2>

      {/* START ANALYSIS */}

      <div className="card">
        <h3>Start New Analysis</h3>

        <div className="form-group">
          <label>Model ID</label>

          <input
            type="text"
            value={modelId}
            onChange={(e) =>
              setModelId(e.target.value)
            }
            placeholder="Enter uploaded model ID"
          />
        </div>

        <div className="form-group">
          <label>Dataset ID</label>

          <input
            type="text"
            value={datasetId}
            onChange={(e) =>
              setDatasetId(e.target.value)
            }
            placeholder="Enter uploaded dataset ID"
          />
        </div>

        <button
          className="btn btn-primary"
          onClick={runAnalysis}
          disabled={
            !modelId || !datasetId || running
          }
        >
          {running
            ? 'Running Analysis...'
            : 'Run Analysis'}
        </button>
      </div>

      {/* PIPELINE */}

      {showPipeline && (
        <div className="card">
          <h3>XAIFA Analysis Pipeline</h3>

          {loading ? (
            <div className="loading">
              Loading pipeline...
            </div>
          ) : (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '0.75rem',
              }}
            >
              {steps.map((step) => (
                <div
                  key={step.order}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '1rem',
                    padding: '0.9rem',
                    background:
                      'var(--background)',
                    borderRadius: '8px',
                    border:
                      '1px solid var(--border)',
                  }}
                >
                  <div
                    style={{
                      width: '28px',
                      height: '28px',
                      borderRadius: '50%',
                      background:
                        step.status ===
                        'completed'
                          ? 'var(--success)'
                          : '#d1d5db',
                      color: 'white',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent:
                        'center',
                      fontSize: '0.8rem',
                      fontWeight: 700,
                    }}
                  >
                    {step.order}
                  </div>

                  <div
                    style={{ fontWeight: 500 }}
                  >
                    {step.name}
                  </div>

                  <div
                    style={{
                      marginLeft: 'auto',
                    }}
                  >
                    <span
                      className={`badge badge-${
                        step.status ===
                        'completed'
                          ? 'success'
                          : 'warning'
                      }`}
                    >
                      {step.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* PIPELINE EXECUTION */}

      {result && (
        <div className="card">
          <h3>Pipeline Execution</h3>

          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '0.75rem',
              marginTop: '1rem',
            }}
          >
            <div
              style={{
                padding: '1rem',
                background: '#dcfce7',
                borderRadius: '8px',
                border:
                  '1px solid #86efac',
              }}
            >
              ✅ train_model.py completed
            </div>

            <div
              style={{
                padding: '1rem',
                background: '#dcfce7',
                borderRadius: '8px',
                border:
                  '1px solid #86efac',
              }}
            >
              ✅ extract_failures.py
              completed
            </div>

            <div
              style={{
                padding: '1rem',
                background: '#dcfce7',
                borderRadius: '8px',
                border:
                  '1px solid #86efac',
              }}
            >
              ✅ run_experiments.py
              completed
            </div>
          </div>
        </div>
      )}

      {/* VISUAL RESULTS */}

      {result && (
        <div className="card">
          <h3>Failure Explanations</h3>

          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '2rem',
            }}
          >
            <div>
              <h4>Failure Dataset Samples</h4>

              <img
                src={`http://127.0.0.1:8001/results/failure_grid.png?t=${Date.now()}`}
                alt="Failure Grid"
                style={{
                  width: '100%',
                  borderRadius: '12px',
                }}
              />
            </div>

            <div>
              <h4>
                GradCAM, SHAP, LIME and
                Combined Methods
              </h4>

              <img
                src={`http://127.0.0.1:8001/results/all_7_methods.png?t=${Date.now()}`}
                alt="All Methods"
                style={{
                  width: '100%',
                  borderRadius: '12px',
                }}
              />
            </div>

            <div>
              <h4>
                Best Combination vs All
                Methods
              </h4>

              <img
                src={`http://127.0.0.1:8001/results/best_vs_all.png?t=${Date.now()}`}
                alt="Best vs All"
                style={{
                  width: '100%',
                  borderRadius: '12px',
                }}
              />
            </div>

            <div>
              <h4>
                PCA Cluster Visualization
              </h4>

              <img
                src={`http://127.0.0.1:8001/results/Combined_pca.png?t=${Date.now()}`}
                alt="PCA"
                style={{
                  width: '100%',
                  borderRadius: '12px',
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* CSV RESULTS TABLE */}

      {csvData.length > 0 && (
        <div className="card">
          <h3>Final Comparison Results</h3>

          <div
            style={{
              overflowX: 'auto',
              marginTop: '1rem',
            }}
          >
            <table
              style={{
                width: '100%',
                borderCollapse: 'collapse',
              }}
            >
              <thead>
                <tr>
                  {csvData[0].map(
                    (header, index) => (
                      <th
                        key={index}
                        style={{
                          border:
                            '1px solid #ccc',
                          padding: '12px',
                          background:
                            '#f3f4f6',
                          textAlign: 'left',
                        }}
                      >
                        {header}
                      </th>
                    )
                  )}
                </tr>
              </thead>

              <tbody>
                {csvData
                  .slice(1)
                  .map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      {row.map(
                        (
                          cell,
                          cellIndex
                        ) => (
                          <td
                            key={cellIndex}
                            style={{
                              border:
                                '1px solid #ccc',
                              padding: '10px',
                            }}
                          >
                            {cell}
                          </td>
                        )
                      )}
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

export default Analysis