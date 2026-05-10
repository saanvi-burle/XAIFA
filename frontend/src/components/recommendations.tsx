import { useEffect, useState } from 'react'

function Recommendations() {

  const [recommendations, setRecommendations] =
    useState('')

  const [tableData, setTableData] =
    useState<string[][]>([])

  const [loading, setLoading] =
    useState(true)

  // =====================================
  // LOAD TXT RECOMMENDATIONS
  // =====================================
  const loadRecommendations = async () => {

    try {

      const response = await fetch(
        `http://127.0.0.1:8001/results/recommendations.txt?t=${Date.now()}`
      )

      const text = await response.text()

      setRecommendations(text)

    } catch (err) {

      console.error(
        'Recommendation load failed',
        err
      )
    }
  }

  // =====================================
  // LOAD CSV FEATURES
  // =====================================
  const loadCSV = async () => {

    try {

      const response = await fetch(
        `http://127.0.0.1:8001/results/recommendation_features.csv?t=${Date.now()}`
      )

      const text = await response.text()

      const rows = text
        .trim()
        .split('\n')
        .map((row) => row.split(','))

      setTableData(rows)

    } catch (err) {

      console.error(
        'CSV load failed',
        err
      )
    }
  }

  // =====================================
  // INITIAL LOAD
  // =====================================
  useEffect(() => {

    const loadAll = async () => {

      await loadRecommendations()

      await loadCSV()

      setLoading(false)
    }

    loadAll()

  }, [])

  // =====================================
  // UI
  // =====================================
  return (

    <div>

      <h2
        style={{
          marginBottom: '1.5rem'
        }}
      >
        XAIFA Recommendations
      </h2>

      {/* LOADING */}

      {loading && (
        <div className="card">
          Loading recommendations...
        </div>
      )}

      {/* RECOMMENDATION TEXT */}

      {!loading && recommendations && (

        <div className="card">

          <h3>
            Adaptive Recommendations
          </h3>

          <pre
            style={{
              whiteSpace: 'pre-wrap',
              lineHeight: 1.8,
              fontSize: '0.95rem',
              background: '#f9fafb',
              padding: '1rem',
              borderRadius: '10px',
              overflowX: 'auto',
            }}
          >
            {recommendations}
          </pre>

        </div>
      )}

      {/* FEATURE TABLE */}

      {!loading &&
        tableData.length > 0 && (

        <div className="card">

          <h3>
            Recommendation Features
          </h3>

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

                  {tableData[0].map(
                    (header, index) => (

                    <th
                      key={index}
                      style={{
                        border:
                          '1px solid #d1d5db',
                        padding: '12px',
                        background:
                          '#f3f4f6',
                        textAlign: 'left',
                      }}
                    >
                      {header}
                    </th>

                  ))}
                </tr>

              </thead>

              <tbody>

                {tableData
                  .slice(1)
                  .map((row, rowIndex) => (

                  <tr key={rowIndex}>

                    {row.map(
                      (cell, cellIndex) => (

                      <td
                        key={cellIndex}
                        style={{
                          border:
                            '1px solid #e5e7eb',
                          padding: '10px',
                        }}
                      >
                        {cell}
                      </td>

                    ))}
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

export default Recommendations