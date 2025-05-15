import { Routes, Route } from 'react-router-dom';
import PowerBIDashboard from './components/PowerBIDashboard';

function App() {
  return (
    <Routes>
      <Route path="/dashboard" element={<PowerBIDashboard />} />
      {/* Other routes */}
    </Routes>
  );
}