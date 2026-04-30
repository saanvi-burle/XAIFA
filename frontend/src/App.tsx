import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import Dashboard from './components/Dashboard'
import Upload from './components/Upload'
import Analysis from './components/Analysis'
import Failures from './components/Failures'

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <header className="header">
          <h1>XAIFA</h1>
          <nav>
            <Link to="/">Dashboard</Link>
            <Link to="/upload">Upload</Link>
            <Link to="/analysis">Analysis</Link>
            <Link to="/failures">Failures</Link>
          </nav>
        </header>
        <main className="main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/failures" element={<Failures />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App