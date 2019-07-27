import React from 'react';
import logo from './logo.svg';
import './App.css';
import {INDI, INDIPanel} from './components/indi';
import ImageReloader from './components/imagereloader';
import ZoomImage from './components/zoomimage';
import INDIClickImage from './components/indiclickimage';
import INDIChartContext from './components/indichartcontext';
import INDIChart from './components/indichart';
import FocusChart from './components/focuschart';
import HistogramChart from './components/histogramchart';

class App extends React.Component {
  constructor(props) {
    super(props);

  }

  render() {
    return (
      <INDI wsurl='websocket' history={ {'Guider': {'offset': 1000, "guider_ra_move": 1000, "guider_dec_move": 1000}, 'Navigator': {'full_res': 10}} }>
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
              <INDIChartContext tab="Guider:Guider" path="Guider:offset" history={true}><INDIChart/></INDIChartContext>
              <INDIChartContext tab="Guider:Guider" path="Guider:guider_ra_move" history={true}><INDIChart/></INDIChartContext>
              <INDIChartContext tab="Guider:Guider" path="Guider:guider_dec_move" history={true}><INDIChart/></INDIChartContext>
              <INDIChartContext tab="Navigator:Focuser" path="Navigator:focus_data:focus_data" enable_blob={true}><FocusChart/></INDIChartContext>
              <INDIChartContext tab="Navigator:FullRes" path="Navigator:full_res" history={true}><INDIChart/></INDIChartContext>
              <INDIChartContext tab="Navigator:FullRes" path="Navigator:histogram:histogram" enable_blob={true}><HistogramChart options={{yaxis: {logarithmic: true}}}/></INDIChartContext>
            </INDIPanel>
          </div>
        </div>
      </INDI>
    );
  }
}

export default App;
