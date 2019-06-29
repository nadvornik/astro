import React from 'react';
import logo from './logo.svg';
import './App.css';
import INDI from './components/indi';
import ImageReloader from './components/imagereloader';

function App() {
  return (
    <div className="App">
      <div className="Images">
        <div className='viewport' >
          <ImageReloader src='navigator.jpg'/>
        </div>
        <div className='viewport' >
          <ImageReloader src='guider.jpg'/>
        </div>
        <div className='viewport' >
          <ImageReloader src='full_res.jpg'/>
        </div>
        <div className='viewport' >
          <ImageReloader src='polar.jpg'/>
        </div>
      </div>
      <div className="INDIwrapper">
        <INDI wsurl='websocket'/>
      </div>
    </div>
  );
}

export default App;
