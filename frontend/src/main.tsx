import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './styles/globals.css'
import App from './App.tsx'

async function enableMocking() {
  if (import.meta.env.MODE !== 'development') {
    return
  }

  try {
    const { worker } = await import('./mocks/browser')
    
    await worker.start({
      onUnhandledRequest: 'bypass',
    })
    if (import.meta.env.DEV) {
      console.log('MSW worker started successfully')
    }
  } catch (error) {
    console.warn('Failed to start MSW worker:', error)
    // Continue anyway - app should work without mocks
  }
}

// Start MSW and render app using top-level await
await enableMocking()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
