import { useState } from 'react';
import axios from '../services/api';

export default function DocumentAnalysis() {
  const [file, setFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/doc/forgery', formData);
      setAnalysis(response.data);
    } catch (error) {
      console.error('Analysis error:', error);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files[0])} />
        <button type="submit">Check Authenticity</button>
      </form>

      {analysis && (
        <div>
          <img src={analysis.ela_analysis} alt="ELA Analysis" />
          <p>Risk Score: {analysis.note}</p>
        </div>
      )}
    </div>
  );
}