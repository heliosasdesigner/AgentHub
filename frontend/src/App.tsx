import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/layout'
import { Home } from './pages/Home'
import { SystemPrompts } from './pages/SystemPrompts'
import './App.css'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="prompts" element={<SystemPrompts />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
