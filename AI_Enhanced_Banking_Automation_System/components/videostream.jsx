import { useEffect, useState } from 'react';

export default function VideoStream() {
  const [streaming, setStreaming] = useState(false);

  const startStream = async () => {
    await fetch('/onboarding/start', { method: 'POST' });
    setStreaming(true);
  };

  return (
    <div>
      <button onClick={startStream}>Start Verification</button>
      {streaming && (
        <img
          src="/onboarding/stream"
          alt="Live Feed"
          style={{ width: '640px', height: '480px' }}
        />
      )}
    </div>
  );
}