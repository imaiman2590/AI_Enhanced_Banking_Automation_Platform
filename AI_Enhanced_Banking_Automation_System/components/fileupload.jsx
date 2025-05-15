import { useState } from 'react';
import axios from '../services/api';

export default function FileUpload() {
  const [file, setFile] = useState(null);
  const [task, setTask] = useState('lending');
  const [results, setResults] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('file', file);
    formData.append('task', task);

    try {
      const response = await axios.post('/process', formData, {
        headers: {'Content-Type': 'multipart/form-data'}
      });
      setResults(response.data);
    } catch (error) {
      console.error('Processing error:', error);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <select value={task} onChange={(e) => setTask(e.target.value)}>
          <option value="lending">Credit Scoring</option>
          <option value="acquisition">Lead Scoring</option>
          <option value="fraud">Fraud Detection</option>
        </select>
        <button type="submit">Analyze</button>
      </form>

      {results && (
        <div className="results">
          <h3>Results</h3>
          <pre>{JSON.stringify(results, null, 2)}</pre>
          {results.plot && (
            <img src={results.plot} alt="Analysis Visualization" />
          )}
        </div>
      )}
    </div>
  );
}