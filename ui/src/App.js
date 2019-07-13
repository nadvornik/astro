import React from 'react';
import logo from './logo.svg';
import './App.css';
import {INDI, INDIPanel} from './components/indi';
import ImageReloader from './components/imagereloader';
import ZoomImage from './components/zoomimage';
import INDIClickImage from './components/indiclickimage';
import INDIChart from './components/indichart';
import FocusChart from './components/focuschart';

class App extends React.Component {
  constructor(props) {
    super(props);

  }

  render() {
    return (
      <INDI wsurl='websocket' history={ {'Guider': {'offset': 1000, "guider_ra_move": 1000, "guider_dec_move": 1000}}  }>
        <div className="App">
          <div className="Images">
            <div className='viewport' >
              <ImageReloader src='navigator.jpg'>
                <INDIClickImage device="Navigator" property="zoom_pos"/>
              </ImageReloader>
            </div>
            <div className='viewport' >
              <ImageReloader src='guider.jpg'>
                <ZoomImage/>
              </ImageReloader>
            </div>
            <div className='viewport' >
              <ImageReloader src='full_res.jpg'>
                <ZoomImage/>
              </ImageReloader>
            </div>
            <div className='viewport' >
              <ImageReloader src='polar.jpg'>
                <ZoomImage/>
              </ImageReloader>
            </div>
          </div>
          <div className="INDIwrapper">
            <INDIPanel>
              <INDIChart tab="Guider:Guider" path="Guider:offset"/>
              <INDIChart tab="Guider:Guider" path="Guider:guider_ra_move"/>
              <INDIChart tab="Guider:Guider" path="Guider:guider_dec_move"/>
              <FocusChart tab="Navigator:Focuser" path="Navigator:focus_data:focus_data"/>
            </INDIPanel>
          </div>
        </div>
      </INDI>
    );
  }
}

export default App;
