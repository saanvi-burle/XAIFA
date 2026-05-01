import { useState } from 'react'
import axios from 'axios'

function Upload() {
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [datasetFile, setDatasetFile] = useState<File | null>(null)
  const [labels, setLabels] = useState('')
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState('')

  const handleModelUpload = async () => {
    if (!modelFile) return
    setUploading(true)
    const formData = new FormData()

    formData.append('model_file', modelFile)

    // ✅ FIXED (case-safe)
    formData.append(
      'model_format',
      modelFile.name.toLowerCase().endsWith('.pt') ? 'torchscript' : 'pytorch_state_dict'
      
    )

    formData.append('architecture', 'resnet18')
    formData.append('input_width', '224')
    formData.append('input_height', '224')
    formData.append('channels', '3')
    formData.append('num_classes', '2')

    try {
      const res = await axios.post('/api/models/upload', formData)
      setMessage(`Model uploaded: ${res.data.model_id}`)
    } catch (err: any) {
      const errorMsg =
        err.response?.data?.detail
          ? JSON.stringify(err.response.data.detail)
          : err.message

      setMessage(`Error: ${errorMsg}`)
    }

    setUploading(false)
  }

  const handleDatasetUpload = async () => {
    if (!datasetFile) return
    setUploading(true)
    const formData = new FormData()

    formData.append('dataset_file', datasetFile)

    formData.append(
      'dataset_format',
      datasetFile.name.endsWith('.csv') ? 'csv_zip' : 'folder_zip'
    )

    try {
      const res = await axios.post('/api/datasets/upload', formData)
      setMessage(`Dataset uploaded: ${res.data.dataset_id}`)
    } catch (err: any) {
      const errorMsg =
        err.response?.data?.detail
          ? JSON.stringify(err.response.data.detail)
          : err.message

      setMessage(`Error: ${errorMsg}`)
    }

    setUploading(false)
  }

  return (
    <div>
      <h2 style={{ marginBottom: '1.5rem' }}>Upload Model & Dataset</h2>
      
      <div className="card">
        <h3>Upload Model</h3>
        <p style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>
          Upload a trained PyTorch model (.pt or .pth)
        </p>
        <div className="form-group">
          <input 
            type="file" 
            accept=".pt,.pth"
            onChange={e => setModelFile(e.target.files?.[0] || null)}
          />
        </div>
        <button 
          className="btn btn-primary" 
          onClick={handleModelUpload}
          disabled={!modelFile || uploading}
        >
          {uploading ? 'Uploading...' : 'Upload Model'}
        </button>
      </div>

      <div className="card">
        <h3>Upload Dataset</h3>
        <p style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>
          Upload a zipped folder dataset or CSV file
        </p>
        <div className="form-group">
          <input 
            type="file" 
            accept=".zip,.csv"
            onChange={e => setDatasetFile(e.target.files?.[0] || null)}
          />
        </div>
        <div className="form-group">
          <label>Class Labels (one per line)</label>
          <textarea
            value={labels}
            onChange={e => setLabels(e.target.value)}
            placeholder="cat&#10;dog"
            rows={4}
            style={{ width: '100%', padding: '0.625rem', border: '1px solid var(--border)', borderRadius: '6px' }}
          />
        </div>
        <button 
          className="btn btn-primary" 
          onClick={handleDatasetUpload}
          disabled={!datasetFile || uploading}
        >
          {uploading ? 'Uploading...' : 'Upload Dataset'}
        </button>
      </div>

      {message && (
        <div className="card" style={{ background: message.startsWith('Error') ? '#fee2e2' : '#dcfce7' }}>
          {message}
        </div>
      )}
    </div>
  )
}

export default Upload