import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [workout, setWorkout] = useState('');

  const generateWorkout = async () => {
    try {
      const response = await axios.get('http://localhost:3001/generate-workout');
      setWorkout(response.data.workout);
    } catch (error) {
      console.error('Error generating workout:', error);
    }
  };

  return (
    <div className="App">
      <h1>Fitness Tracker</h1>
      <button onClick={generateWorkout}>Generate Workout</button>
      {workout && <div><h2>Generated Workout:</h2><p>{workout}</p></div>}
    </div>
  );
}

export default App;
